import torch
from tqdm import tqdm
from lora_decomp import LoRaDecomp
from model_specific_utils import get_linears_llama

@torch.no_grad()
def get_lora_decomp(M, I):
    rank = min(M.shape)
    U, S, V = torch.svd_lowrank(M@I, q=rank)
    I_inv = torch.linalg.pinv(I, rtol=1e-18, atol=0)
    V = ((V.T)@I_inv).T
    return U[:, :rank].cpu(), S[:rank].cpu(), V[:, :rank].cpu()

def _get_initial_lora_decomp(module, inp_tens, lora_decomp_func=get_lora_decomp):
    torch.cuda.empty_cache()
    usv = lora_decomp_func(module.weight, inp_tens.T.contiguous().cuda())
    lora_decomp = LoRaDecomp(module, Q=usv[0], B=usv[1]*usv[2], X=inp_tens.T.contiguous())
    torch.cuda.empty_cache()
    return lora_decomp

@torch.no_grad()
def get_initial_lora_decomps(model, inp_tensors, get_linears=get_linears_llama, lora_decomp_func=get_lora_decomp, get_initial_lora_decomp_func=_get_initial_lora_decomp):
    torch.manual_seed(42)
    lora_decomps = {}
    linears = get_linears(model)
    assert len(inp_tensors) == len(linears)
    with torch.no_grad():
        for layer_i, layers in tqdm(enumerate(linears)):
            entries = []
            for inp_tens, module in zip(inp_tensors[layer_i], layers):
                lora_decomp = get_initial_lora_decomp_func(module, inp_tens, lora_decomp_func=lora_decomp_func)
                entries.append(lora_decomp)

            lora_decomps[layer_i] = entries
    return lora_decomps
