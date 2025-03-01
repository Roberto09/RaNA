import argparse
import os
import pandas as pd
import torch
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_specific_utils import replace_qkv_layers_llama
from basis_learners import *
from input_samples import *
from input_samples import _get_input_tensors
from rank_adaptive_experts import *
from dataset import get_baseline_dataset_v2
from rank_adaptive import *
from model_specific_utils import get_mlps_llama
from basis_learners import _get_initial_lora_decomp

MODEL_REVISIONS = {
    "meta-llama/Llama-2-7b-hf": "01c7f73d771dfac7d292323805ebc428287df4f9",
    "google/gemma-2b": "68e273d91b1d6ea57c9e6024c4f887832f7b43fa",
}

def genereate_rana(PRUNE_RATIO, MODEL, DEVICE, DATASET_FILENAME, SAVE_FILENAME, PRE_LOAD_MODEL_STATE_DICT):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{DEVICE}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISIONS[MODEL])
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISIONS[MODEL])
    if PRE_LOAD_MODEL_STATE_DICT is not None:
        model.load_state_dict(torch.torch.load(PRE_LOAD_MODEL_STATE_DICT))

    is_llama = MODEL == "meta-llama/Llama-2-7b-hf"
    if is_llama:
        model = model.half()
        replace_qkv_layers_llama(model)

    dataset = get_baseline_dataset_v2("togethercomputer/RedPajama-Data-1T-Sample", f"{DATASET_FILENAME}.pkl", train_size=100_000)
    train_data, eval_data = dataset["train"], dataset["test"]

    ttl_tokens_mult = 8
    inp_sample_size = 32_000
    seq_size = 1024
    tokenized_data = get_tokenized_train_data(train_data.shuffle(seed=123), tokenizer, ttl_tokens=ttl_tokens_mult*inp_sample_size, seq_size=seq_size)

    def custom_get_linears_llama(m):
        linear_layers = []
        for layer in model.model.layers:
            attn = layer.self_attn
            mlp = layer.mlp
            linear_layers.append([attn.q_proj.stacked_qkv.qkv, mlp.up_proj, mlp.gate_proj])
        return linear_layers
    
    def custom_get_linears_gemma(m):
        linear_layers = []
        for layer in model.model.layers:
            mlp = layer.mlp
            linear_layers.append([mlp.up_proj, mlp.gate_proj])
        return linear_layers
    
    custom_get_linears = custom_get_linears_llama if is_llama else custom_get_linears_gemma

    input_tensors = _get_input_tensors(model, tokenized_data, inp_sample_size=inp_sample_size,
                                     ttl_tokens_mult=ttl_tokens_mult, seq_size=seq_size, mem_efficient=True, get_linears=custom_get_linears)

    def _float_32_get_initial_lora_decomp(module, inp_tens, lora_decomp_func=get_lora_decomp):
        torch.cuda.empty_cache()
        usv = lora_decomp_func(module.weight.to(torch.float32), inp_tens.T.contiguous().to(torch.float32).cuda())
        lora_decomp = LoRaDecomp(module, Q=usv[0].to(torch.float16), B=(usv[1]*usv[2]).to(torch.float16), X=inp_tens.T.contiguous())
        torch.cuda.empty_cache()
        return lora_decomp

    get_initial_lora_decomp_func = _float_32_get_initial_lora_decomp if is_llama else _get_initial_lora_decomp

    lora_decomps = get_initial_lora_decomps(model, input_tensors, custom_get_linears, get_initial_lora_decomp_func=get_initial_lora_decomp_func)
    lora_decomps = [lora_decomps[i] for i in range(len(lora_decomps))]

    mlps = get_mlps_llama(model)
    if is_llama:
        qkvs = [layer.self_attn.q_proj.stacked_qkv.qkv for layer in model.model.layers]
        
        qkv_decomps = [lora_decomps[i][0] for i in range(len(lora_decomps))]
        up_proj_decomps = [lora_decomps[i][1] for i in range(len(lora_decomps))]
        gate_proj_decomps = [lora_decomps[i][2] for i in range(len(lora_decomps))]
        
        up_proj_inps = [input_tensors[i][1] for i in range(len(input_tensors))]
        qkv_inps = [input_tensors[i][0] for i in range(len(input_tensors))]
       
        mlp_outs = []
        qkv_outs = []
    else:
        up_proj_decomps = [lora_decomps[i][0] for i in range(len(lora_decomps))]
        gate_proj_decomps = [lora_decomps[i][1] for i in range(len(lora_decomps))]
        
        up_proj_inps = [input_tensors[i][0] for i in range(len(input_tensors))]
        mlp_outs = []

    model.eval()

    with torch.no_grad():
        if is_llama:
            assert len(qkvs) == len(qkv_inps)
            for qkv, qkv_inp in zip(qkvs, qkv_inps):
                qkv_outs.append(qkv(qkv_inp.cuda()).detach().cpu())
        assert len(mlps) == len(up_proj_inps)
        for mlp, up_inp in zip(mlps, up_proj_inps):
            mlp_outs.append(mlp(up_inp.cuda()).detach().cpu())
    model.train()

    import gc; gc.collect()
    torch.cuda.empty_cache()

    if is_llama:
        print("Computing RaNA QKVs ...")
        assert len(qkvs) == len(qkv_outs) == len(qkv_decomps)
        rank_adaptive_qkvs = []
        for orig_qkv, qkv_out, qkv_proj_decomp in zip(qkvs, qkv_outs, qkv_decomps):
            orig_qkv_32 = orig_qkv.float()
            qkv_out_32 = qkv_out.to(torch.float32)
            qkv_proj_decomp_32 = LoRaDecomp(qkv_proj_decomp.module.float(), qkv_proj_decomp.Q.to(torch.float32), qkv_proj_decomp.B.to(torch.float32), qkv_proj_decomp.X.to(torch.float32), out=(None,))
            
            rank_adaptive_qkvs.append(get_best_rank_adaptive_qkv(orig_qkv_32, qkv_proj_decomp_32, PRUNE_RATIO))
            
            orig_qkv_32.half()
            qkv_proj_decomp_32.module.half()

        print("Computing RaNA MLPs ...")
        assert len(mlps) == len(mlp_outs) == len(up_proj_decomps) == len(gate_proj_decomps)
        rank_adaptive_mlps = []
        for orig_mlp, mlp_out, up_proj_decomp, gate_proj_decomp in zip(mlps, mlp_outs, up_proj_decomps, gate_proj_decomps):
            orig_mlp_32 = orig_mlp.float()
            mlp_out_32 = mlp_out.to(torch.float32)
            up_proj_decomp_32 = LoRaDecomp(up_proj_decomp.module.float(), up_proj_decomp.Q.to(torch.float32), up_proj_decomp.B.to(torch.float32), up_proj_decomp.X.to(torch.float32), out=(None,))
            gate_proj_decomp_32 = LoRaDecomp(gate_proj_decomp.module.float(), gate_proj_decomp.Q.to(torch.float32), gate_proj_decomp.B.to(torch.float32), gate_proj_decomp.X.to(torch.float32), out=(None,))
            
            rank_adaptive_mlps.append(get_best_rank_adaptive_mlp_up_down_gate(orig_mlp_32, up_proj_decomp_32, gate_proj_decomp_32, PRUNE_RATIO, mlp_out_32, ret_descr_ratio=True))

            orig_mlp_32.half()
            up_proj_decomp_32.module.half()
            gate_proj_decomp_32.module.half()
    else:
        print("Computing RaNA MLPs ...")
        assert len(mlps) == len(mlp_outs) == len(up_proj_decomps) == len(gate_proj_decomps)
        rank_adaptive_mlps = []
        for orig_mlp, mlp_out, up_proj_decomp, gate_proj_decomp in zip(mlps, mlp_outs, up_proj_decomps, gate_proj_decomps):
            rank_adaptive_mlps.append(get_best_rank_adaptive_mlp_up_down_gate(orig_mlp, up_proj_decomp, gate_proj_decomp, PRUNE_RATIO, mlp_out, ret_descr_ratio=True))

    for mlp in rank_adaptive_mlps:
        mlp[0].down_proj = None

    os.makedirs("./rana_models/", exist_ok=True)
    
    pd.to_pickle(rank_adaptive_mlps, f"./rana_models/{SAVE_FILENAME}_{PRUNE_RATIO}_mlps.pkl")
    if is_llama:
        pd.to_pickle(rank_adaptive_qkvs, f"./rana_models/{SAVE_FILENAME}_{PRUNE_RATIO}_qkvs.pkl")

    print("DONE GENERATING RANA")

def main():
    parser = argparse.ArgumentParser(description="generate_rana_model")

    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", type=str, help="Model Name")
    parser.add_argument("--prune_ratio", type=float, help="Prune Ratio", required=True)
    parser.add_argument("--device", default=0, type=int, help="CUDA Device ID")
    parser.add_argument("--dataset_filename", default="red_pajama_dataset", type=str, help="dataset file name (where to save / where to load from the dataset")
    parser.add_argument("--save_filename", default="rana_non_ft", type=str, help="model file name (where to save the rana model)")
    parser.add_argument("--pre-load-model-state-dict", default=None, type=str, help="State-dict to load into model before generating rana adapters")

    args = parser.parse_args()

    model = args.model
    prune_ratio = args.prune_ratio
    device = args.device
    dataset_filename = args.dataset_filename
    save_filename = args.save_filename
    pre_load_model_state_dict = args.pre_load_model_state_dict
    
    assert model in MODEL_REVISIONS, f"MODEL: {model} not supported"

    genereate_rana(prune_ratio, model, device, dataset_filename, save_filename, pre_load_model_state_dict)

if __name__ == "__main__":
    main()
