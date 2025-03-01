import torch
from torch import nn
from rank_adaptive import MaskedLinear, ThreshedMLP
from copy import deepcopy
import triton
import triton.language as tl
from torch.nn.functional import silu
import torch.nn as nn
from functools import partial

# Adapted from: https://github.com/ScalingIntelligence/CATS.
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 2048}, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 2048}, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 2048}, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 2048}, num_warps=8
        ),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def custom_gather_transposed_gemv_flag_atomicadd_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # Stride variables
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr, # assumes batch size is 1
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)

    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm_mask = rm < M
    rn_mask = rn < N

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm_mask, other=0) > 0

    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rm
    Y = Y + rn

    a = tl.load(A, mask=idx[:, None] & rn_mask[None, :], other=0.0)
    x0 = tl.load(X, mask=rm_mask, other=0.0)
    acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    tl.atomic_add(Y, acc0, mask=rn_mask)


def compute_grid(META, Z, N):
    return (
        triton.cdiv(Z, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

# Adapted from: https://github.com/ScalingIntelligence/CATS.
@torch.jit.ignore
def custom_kernel_gather_transposed_gemv_flag_3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = weight[idx, :]^T @ x.
    :param x: input tensor
    :param weight: weight matrix
    :param idx: indices
    :return: result tensor
    """
    Z, N = weight.shape
    beam_width, seq_len, _ = x.shape

    grid = partial(compute_grid, Z=Z, N=N)

    kernel = custom_gather_transposed_gemv_flag_atomicadd_kernel
    kernel[grid](
        output,  # data ptrs
        weight,
        x,
        idx,
        Z,  # shapes
        N,
        Z // 128,  # key for triton cache (limit number of compilations)
        N // 32,
        weight.stride(0),  # strides
        beam_width,  # can't use kwargs because auto-tuner requires args
    )
    return output

@torch.jit.script
def abs_mul_leq(x: torch.Tensor, W_norm:torch.Tensor, thresh:torch.Tensor):
    return (torch.abs(x) * W_norm) >= thresh

class MaskedInputLinearKernel(nn.Module):
    def __init__(self, down_proj : nn.Linear, thresh : float):
        super().__init__()
        orig_W = down_proj.weight.data
        self.W_t = nn.Parameter(orig_W.clone().T.contiguous())
        self.W_norm = nn.Parameter(torch.linalg.norm(orig_W, dim=0).detach())
        self.thresh = nn.Parameter(torch.tensor(thresh))
    
    def forward(self, x, out=None):
        m = abs_mul_leq(x, self.W_norm, self.thresh)
        res = custom_kernel_gather_transposed_gemv_flag_3d(x, self.W_t, m, out)
        return res

@torch.jit.script
def silu_scaled(x, up_proj_x):
    return silu(x) * up_proj_x

class RaNAMLPKernelFast(nn.Module):
    def __init__(self, mlp : ThreshedMLP):
        super().__init__()
        up_proj = mlp.up_proj
        gate_proj = mlp.activ.gate_proj

        orig_B_up, thresh_up = up_proj.B, up_proj.thresh
        orig_B_gate, thresh_gate = gate_proj.B, gate_proj.thresh
        self.B = nn.Linear(orig_B_up.in_features, orig_B_up.out_features + orig_B_gate.out_features, bias=False)
        self.B.weight.data = torch.concatenate((orig_B_up.weight.data, orig_B_gate.weight.data)).detach()
        self.thresh_bx = nn.Parameter(torch.concat([torch.tensor([thresh_up]*orig_B_up.out_features), torch.tensor([thresh_gate]*orig_B_gate.out_features)]))
        self.index_limiter_bx = orig_B_up.out_features

        orig_A_up = up_proj.A
        self.A_t_up = nn.Parameter(orig_A_up.weight.data.clone().T.contiguous())

        orig_A_gate = gate_proj.A
        self.A_t_gate = nn.Parameter(orig_A_gate.weight.data.clone().T.contiguous())

        self.down_proj = MaskedInputLinearKernel(mlp.down_proj, mlp.thresh)

        self.out_idx1, self.out_idx2, self.out_idx3 = self.A_t_up.shape[1], self.A_t_up.shape[1]*2, self.A_t_up.shape[1]*2+self.down_proj.W_t.shape[1]

    def get_bx(self, x):
        bx = self.B(x)
        bx_m = bx * bx >= self.thresh_bx
        bx_up = bx[:, :, :self.index_limiter_bx]
        bx_m_up = bx_m[:, :, :self.index_limiter_bx]
        
        bx_gate = bx[:, :, self.index_limiter_bx:]
        bx_m_gate = bx_m[:, :, self.index_limiter_bx:]
        return bx_up, bx_m_up, bx_gate, bx_m_gate
    

    def forward(self, x):
        out = torch.zeros(
            1,
            1,
            self.out_idx3,
            device=x.device,
            dtype=torch.float32,
        )
        out1 = out[:, :, :self.out_idx1]
        out2 = out[:, :, self.out_idx1:self.out_idx2]
        out3 = out[:, :, self.out_idx2:]

        bx_up, bx_m_up, bx_gate, bx_m_gate = self.get_bx(x)

        o_up = custom_kernel_gather_transposed_gemv_flag_3d(bx_up, self.A_t_up, bx_m_up, out1)
        o_gate = custom_kernel_gather_transposed_gemv_flag_3d(bx_gate, self.A_t_gate, bx_m_gate, out2)

        o_up_gate = silu_scaled(o_gate, o_up)

        res = self.down_proj(o_up_gate, out3)
        return res


class MaskedRankLinearKernel(nn.Module):
    def __init__(self, masked_linear : MaskedLinear):
        super().__init__()
        orig_A, orig_B, thresh = masked_linear.A, masked_linear.B, masked_linear.thresh
        self.A_t = nn.Parameter(orig_A.weight.data.clone().T.contiguous())
        self.B = deepcopy(orig_B)
        self.thresh = nn.Parameter(torch.tensor(thresh))
    
    def forward(self, x):
        out = torch.zeros(
            1,
            1,
            self.A_t.shape[1],
            device=x.device,
            dtype=torch.float32,
        )
        o1 = self.B(x)
        m = o1 * o1 >= self.thresh
        o2 = custom_kernel_gather_transposed_gemv_flag_3d(o1, self.A_t, m, out)
        return o2
