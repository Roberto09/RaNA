import torch
from torch import nn
import numpy as np
from tqdm import tqdm 
from rank_adaptive_experts import RunningMean
import itertools
import math
from functools import lru_cache 

def get_dynamic_pruning_ratio(mat_dims, effective_prune_ratio, nB=None):
    m, n = mat_dims
    nB = n if nB is None else nB
    target_fps = 2 * (1-effective_prune_ratio) * m * n
    dynamic_prune_ratio = (target_fps - 2 * n * nB) / (2 * m)
    return dynamic_prune_ratio / n

@torch.no_grad()
def get_upperbound_coverage(errors, prune_ratio):
    err_sum = errors.sum()
    to_keep_elements = int(errors.shape[0]*errors.shape[1]*(1-prune_ratio))
    errors = errors.view(-1).contiguous()
    errors = torch.sort(errors, descending=True).values
    coverage = errors[:to_keep_elements].sum()
    return (coverage/err_sum).item()

@torch.no_grad()
def _get_coverage_dk_nB(errors, dk, nB):
    err_sum = errors.sum()
    to_keep_elements = int(errors.shape[0]*errors.shape[1]*dk)
    errors = errors[:nB].view(-1).contiguous()
    assert to_keep_elements < len(errors) and dk > 0
    errors = torch.sort(errors, descending=True).values
    coverage = errors[:to_keep_elements].sum()
    thresh = errors[to_keep_elements]
    return (coverage/err_sum).item(), thresh.item()

@torch.no_grad()
def get_error_coverages(lora_decomp, prune_ratio, skip_size=10):
    mat_dims = lora_decomp.Q.shape
    errors = (lora_decomp.get_errs()**2).cuda()

    coverages = []
    nBs = []
    dks = []
    nB = mat_dims[1]
    for _ in tqdm(range(1000)):
        dk = get_dynamic_pruning_ratio(mat_dims, prune_ratio, nB)
        if dk * mat_dims[1] >= nB: break
        if dk < 0:
            nB -= skip_size
            continue
        coverage, _ = _get_coverage_dk_nB(errors, dk, nB)
        
        coverages.append(coverage)
        nBs.append(nB)
        dks.append(dk)
        
        nB -= skip_size 
    return coverages, nBs, dks, get_upperbound_coverage(errors, prune_ratio)

def get_best_error_coverage(coverages, nBs, dks):
    assert len(coverages) == len(nBs) == len(dks)
    best_idx = np.argmax(coverages)
    return coverages[best_idx], nBs[best_idx], dks[best_idx]

class MaskedLinear(nn.Module):
    def __init__(self, lora_decomp, dk, nB, track_eff_rank=False):
        super().__init__()
        errors = (lora_decomp.get_errs()**2).cuda()
        _, self.thresh = _get_coverage_dk_nB(errors, dk, nB)
        orig_layer = lora_decomp.module
        self.max_flops = 2 * lora_decomp.Q.shape[0] * lora_decomp.Q.shape[1]
        A = lora_decomp.Q[:, :nB].contiguous()
        B = lora_decomp.B[:, :nB].contiguous()
        self.A = nn.Linear(in_features=A.shape[1], out_features=A.shape[0], bias=orig_layer.bias is not None)
        self.B = nn.Linear(in_features=B.shape[0], out_features=B.shape[1], bias=False)
        self.A.weight.data = A.detach().clone()
        if orig_layer.bias is not None:
            self.A.bias.data = orig_layer.bias.data.detach().clone()
        self.B.weight.data = B.T.contiguous().detach().clone()
        self.track_eff_rank = track_eff_rank
        if self.track_eff_rank:
            self.running_effective_rank = RunningMean()

    def maybe_update_effective_rank(self, keep_mask):
        if not self.track_eff_rank: return
        if self.training: return
        with torch.no_grad():
            ttl_to_keep = keep_mask.view(-1, keep_mask.shape[-1]).sum(dim=-1)
            assert keep_mask.shape[0] == 1, "only support bs of 1"
            self.running_effective_rank.update_multiple(ttl_to_keep.cpu())

    def get_effective_flops(self):
        assert self.track_eff_rank
        eff_rank = self.running_effective_rank.get_mean()
        flops = 2 * (self.B.weight.shape[0]*self.B.weight.shape[1] + self.A.weight.shape[0]*eff_rank)
        return flops

    def get_effective_flops_ratio(self):
        return self.get_effective_flops() / self.max_flops

    def forward(self, x):
        bx = self.B(x)
        with torch.no_grad():
            keep_mask = (bx**2 >= self.thresh).to(x)
            self.maybe_update_effective_rank(keep_mask)
        bx = bx * keep_mask
        return self.A(bx)

class ThreshedMLP(nn.Module):
    def __init__(self, up_proj, down_proj, activ, thresh, track_eff_rank=False):
        super().__init__()
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.thresh = thresh
        self.activ = activ
        self.track_eff_rank = track_eff_rank
        if self.track_eff_rank:
            self.running_effective_rank = RunningMean()
    
    def get_effective_flops_ratio(self):
        assert self.track_eff_rank
        mat_dims = self.down_proj.weight.shape
        flops_up_proj = self.up_proj.get_effective_flops()
        flops_down_proj = 2 * min(mat_dims) * self.running_effective_rank.get_mean()
        flops_gate_proj = self.activ.gate_proj.get_effective_flops()
        max_flops = 3*2*mat_dims[0]*mat_dims[1]
        used_flops = (flops_up_proj + flops_down_proj + flops_gate_proj)
        return used_flops / max_flops

    def compute_orig_fc1_out(self, x):
        up_proj_x = self.up_proj(x)
        return self.activ(x, up_proj_x)

    def _get_activation_importances_fc1_o(self, fc1_o):
        with torch.no_grad():
            fc2_norm = torch.linalg.norm(self.down_proj.weight, dim=0)
            orig_shape = fc1_o.shape
            fc1_o = fc1_o.view(-1, orig_shape[-1]).contiguous()
            norms = fc1_o.abs() * fc2_norm
            norms = norms.view(orig_shape).contiguous()
            return norms.detach()

    def _get_activation_importances(self, x):
        with torch.no_grad():
            fc1_o = self.compute_orig_fc1_out(x)
            return self._get_activation_importances_fc1_o(fc1_o)

    def maybe_update_effective_rank(self, keep_mask):
        if not self.track_eff_rank: return
        if self.training: return
        with torch.no_grad():
            ttl_to_keep = keep_mask.view(-1, keep_mask.shape[-1]).sum(dim=-1)
            assert keep_mask.shape[0] == 1, "only support bs of 1"
            self.running_effective_rank.update_multiple(ttl_to_keep.cpu())

    def forward(self, x):
        orig_fc1_out = self.compute_orig_fc1_out(x)
        with torch.no_grad():
            keep_mask = (self._get_activation_importances_fc1_o(orig_fc1_out.detach()) >= self.thresh).to(x)
            self.maybe_update_effective_rank(keep_mask)
        fc1_out = orig_fc1_out * keep_mask
        fc2_out = self.down_proj(fc1_out)
        return fc2_out

class GateActiv(nn.Module):
    def __init__(self, gate_proj, activ):
        super().__init__()
        self.gate_proj = gate_proj
        self.activ = activ

    def forward(self, x, up_proj_x):
        return self.activ(self.gate_proj(x)) * up_proj_x

@lru_cache(maxsize=1)
def _get_perm(size):
    torch.manual_seed(42)
    perm = torch.randperm(size)
    return perm

def get_threshed_mlp_up_gate_down(mlp, up_proj_decomp, gate_decomp, inp_tens, pr_up, pr_gate, pr_down):
    act_fn = mlp.act_fn
    up_proj = mlp.up_proj
    down_proj = mlp.down_proj
    gate_proj = mlp.gate_proj

    activ = GateActiv(gate_proj=gate_proj, activ=act_fn).cuda()
    threshed_mlp = ThreshedMLP(up_proj, down_proj, activ, None, track_eff_rank=True).cuda()
    eg_activs = threshed_mlp._get_activation_importances(inp_tens).contiguous().view(-1)
    print("sampling threshed mlp activs")
    idxs = _get_perm(eg_activs.shape[0])[:int(eg_activs.shape[0]*0.2)]
    eg_activs = eg_activs[idxs]
    
    import gc; gc.collect()
    torch.cuda.empty_cache()

    eg_activs = torch.sort(eg_activs).values
    to_remove = int(len(eg_activs)*pr_down)
    thresh = eg_activs[to_remove].detach().item()
    del eg_activs
    threshed_mlp.thresh = thresh

    coverages, nBs, dks, ub_coverage = get_error_coverages(up_proj_decomp, pr_up)
    best_error_coverage, best_nB, best_dk = get_best_error_coverage(coverages, nBs, dks)
    masked_up_proj = MaskedLinear(up_proj_decomp, best_dk, best_nB, track_eff_rank=True)
    threshed_mlp.up_proj = masked_up_proj

    coverages, nBs, dks, ub_coverage = get_error_coverages(gate_decomp, pr_gate)
    best_error_coverage, best_nB, best_dk = get_best_error_coverage(coverages, nBs, dks)
    masked_gate = MaskedLinear(gate_decomp, best_dk, best_nB, track_eff_rank=True)
    activ.gate_proj = masked_gate

    return threshed_mlp

def get_best_rank_adaptive_qkv(qkv_linear, qkv_decomp, ttl_prune_ratio):
    coverages, nBs, dks, ub_coverage = get_error_coverages(qkv_decomp, ttl_prune_ratio)
    best_error_coverage, best_nB, best_dk = get_best_error_coverage(coverages, nBs, dks)
    masked_qkv = MaskedLinear(qkv_decomp, best_dk, best_nB, track_eff_rank=True)
    return masked_qkv

@torch.no_grad()
def get_best_rank_adaptive_mlp_up_down_gate(mlp, up_decomp, gate_decomp, ttl_prune_ratio, orig_mlp_out, ret_descr_ratio=False, keep_ratio_options=None):
    ttl_keep_ratio = 1-ttl_prune_ratio
    added_keep_ratio = ttl_keep_ratio * 3
    
    if keep_ratio_options is None:
        keep_ratio_options = [i/18 for i in range(3, 10)]
    keep_ratio_options = list(itertools.product(keep_ratio_options, keep_ratio_options, keep_ratio_options))
    keep_ratio_options = [opt for opt in keep_ratio_options if math.fabs(sum(opt) - 1) <= 0.0001] 

    new_mlps = []
    new_errors = []
    new_distr = []
    inp = up_decomp.X.T.contiguous().cuda()
    for kr_u, kr_d, kr_g in keep_ratio_options:
        kr_u = kr_u * added_keep_ratio
        kr_d = kr_d * added_keep_ratio
        kr_g = kr_g * added_keep_ratio
        new_mlp = get_threshed_mlp_up_gate_down(mlp, up_decomp, gate_decomp, inp, 1-kr_u, 1-kr_g, 1-kr_d).cuda()
        with torch.no_grad():
            new_out = new_mlp(inp)
            out_diff = (torch.linalg.norm(orig_mlp_out.cuda() - new_out)**2).cpu().detach().item()
            new_errors.append(out_diff)
            new_mlps.append(new_mlp.cpu())
            new_distr.append((kr_u, kr_d, kr_g))
    
    best_idx = np.argmin(new_errors)
    best_mlp = new_mlps[best_idx]
    best_error = new_errors[best_idx]
    best_distr = new_distr[best_idx]
    descr_ratio = 1 - (best_error / (torch.linalg.norm(orig_mlp_out)**2)).item()
    if ret_descr_ratio:
        return best_mlp.cpu(), best_distr, descr_ratio, new_errors
    return best_mlp.cpu()