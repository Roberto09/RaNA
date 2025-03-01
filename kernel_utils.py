import torch
from torch import nn
from rank_adaptive import MaskedLinear, ThreshedMLP
from kernel_utils_fast import RaNAMLPKernelFast, silu_scaled, abs_mul_leq, MaskedRankLinearKernel
from tqdm import tqdm

class RaNAMLPKernel(nn.Module):
    def __init__(self, mlp : ThreshedMLP):
        super().__init__()
        self.fast_rana_mlp = RaNAMLPKernelFast(mlp)

    def forward(self, x):
        if x.shape[1] == 1:
            return self.fast_rana_mlp(x)
        
        bx_up, bx_m_up, bx_gate, bx_m_gate = self.fast_rana_mlp.get_bx(x)

        bx_up_squeezed = (bx_up * bx_m_up.to(bx_up)).squeeze(0)
        o_up = bx_up_squeezed @ self.fast_rana_mlp.A_t_up
        o_up = o_up.unsqueeze(0).contiguous()

        bx_gate_squeezed = (bx_gate * bx_m_gate.to(bx_gate)).squeeze(0)
        o_gate = bx_gate_squeezed @ self.fast_rana_mlp.A_t_gate
        o_gate = o_gate.unsqueeze(0).contiguous()

        o_up_gate = silu_scaled(o_gate, o_up).squeeze(0)
        m_down = abs_mul_leq(o_up_gate, self.fast_rana_mlp.down_proj.W_norm, self.fast_rana_mlp.down_proj.thresh)
        o_down = (o_up_gate * m_down).squeeze(0) @ self.fast_rana_mlp.down_proj.W_t
        o_down = o_down.unsqueeze(0).contiguous()
        return o_down

class RaNAQKVKernel(nn.Module):
    def __init__(self, masked_linear : MaskedLinear):
        super().__init__()
        self.fast_rana_qkv = MaskedRankLinearKernel(masked_linear)

    def forward(self, x):
        if x.shape[1] == 1:
            return self.fast_rana_qkv(x)
        bx = self.fast_rana_qkv.B(x)
        m = bx * bx >= self.fast_rana_qkv.thresh
        bx_squeezed = (bx * m).squeeze(0)
        o_qkv = bx_squeezed @ self.fast_rana_qkv.A_t
        o_qkv = o_qkv.unsqueeze(0).contiguous()
        return o_qkv

def get_kernel_rana_model(model, do_mlp=True, do_qkv=True):
    for layer in model.model.layers:
        if do_mlp:
            mlp = RaNAMLPKernel(layer.mlp)
            mlp.fast_rana_mlp = torch.jit.script(mlp.fast_rana_mlp)
            layer.mlp = mlp
        
        if do_qkv:
            attn = layer.self_attn
            qkv = attn.q_proj.stacked_qkv.qkv
            kernel_qkv = RaNAQKVKernel(qkv)
            kernel_qkv.fast_rana_qkv = torch.jit.script(kernel_qkv.fast_rana_qkv)
            attn.q_proj.stacked_qkv.qkv = kernel_qkv
            attn.k_proj.stacked_qkv.qkv = kernel_qkv
            attn.v_proj.stacked_qkv.qkv = kernel_qkv
    return model

@torch.no_grad()
def warmup_rana_kernels(model):
    eg_inp = torch.randn(1, 1, 4096).cuda()
    for layer in tqdm(model.model.layers):
        mlp = layer.mlp
        attn = layer.self_attn
        q,k,v = attn.q_proj, attn.k_proj, attn.v_proj
        o_mlp = mlp(eg_inp)
        o_q = q(eg_inp)
        o_k = k(eg_inp)
        o_v = v(eg_inp)
        torch.cuda.synchronize()
