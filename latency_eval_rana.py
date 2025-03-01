import os
import argparse
import pandas as pd
import torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from model_specific_utils import replace_qkv_layers_llama
from time_evals import benchmark_decode
from custom_sample_funcs import replace_model_generate
from kernel_utils import get_kernel_rana_model, warmup_rana_kernels

MODEL_REVISIONS = {
    "meta-llama/Llama-2-7b-hf": "01c7f73d771dfac7d292323805ebc428287df4f9",
}

def latency_eval_rana(MODEL, DEVICE, DATASET_FILENAME, LOAD_ADAPTERS_FILENAME, LOAD_FT_STATE_DICT_FILENAME, PRUNE_RATIO, LAT_SAVE_DIR, ITERS):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{DEVICE}"

    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISIONS[MODEL])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISIONS[MODEL])
    tokenizer.pad_token = tokenizer.eos_token

    rana_qkv_adapters = pd.read_pickle(f"./rana_models/{LOAD_ADAPTERS_FILENAME}_{PRUNE_RATIO}_qkvs.pkl")
    rana_adapters = pd.read_pickle(f"./rana_models/{LOAD_ADAPTERS_FILENAME}_{PRUNE_RATIO}_mlps.pkl")

    replace_qkv_layers_llama(model)
    for layer, rana_qkv_adapter in zip(model.model.layers, rana_qkv_adapters):
        layer.self_attn.q_proj.stacked_qkv.qkv = rana_qkv_adapter

    for layer, rana_mlp in zip(model.model.layers, rana_adapters):
        rana_mlp[0].down_proj = layer.mlp.down_proj
        layer.mlp = rana_mlp[0]

    model.load_state_dict(torch.load(f"./rana_models/{LOAD_FT_STATE_DICT_FILENAME}_{PRUNE_RATIO}_state_dict_finetuned.pkl"))
    model = get_kernel_rana_model(model)

    model = model.eval()
    model = model.cuda()
    warmup_rana_kernels(model)

    import gc; gc.collect()
    torch.cuda.empty_cache()

    save_dir = f"{LAT_SAVE_DIR}_{PRUNE_RATIO}"
    os.makedirs(save_dir, exist_ok=True)
    model = replace_model_generate(model, save_dir)
    benchmark_decode(model, tokenizer, f"{DATASET_FILENAME}.pkl", iters=ITERS)

def main():
    parser = argparse.ArgumentParser(description="latency_eval_rana")

    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", type=str, help="Model Name")
    parser.add_argument("--prune_ratio", type=float, help="Prune Ratio", required=True)
    parser.add_argument("--device", default=0, type=int, help="CUDA Device ID")
    parser.add_argument("--dataset_filename", default="red_pajama_dataset", type=str, help="dataset file name (where to save / where to load from the dataset")
    parser.add_argument("--load_rana_filename", default="rana_non_ft", type=str, help="filename from which rana adapters should be loaded")
    parser.add_argument("--load_rana_ft_sate_dict_filename", default="rana", type=str, help="filename from which rana-adapted fine-tuned model state dict should be loaded")
    parser.add_argument("--lat_save_dir", default="latency_rana_results", type=str, help="directory where latency results will be saved")
    parser.add_argument("--iters", default=100, type=int, help="CUDA Device ID")

    args = parser.parse_args()

    model = args.model
    prune_ratio = args.prune_ratio
    device = args.device
    dataset_filename = args.dataset_filename
    load_rana_filename = args.load_rana_filename
    load_rana_ft_state_dict_filename = args.load_rana_ft_sate_dict_filename
    lat_save_dir = args.lat_save_dir
    iters = args.iters
    
    latency_eval_rana(model, device, dataset_filename, load_rana_filename, load_rana_ft_state_dict_filename, prune_ratio, lat_save_dir, iters)

    assert model in MODEL_REVISIONS, f"MODEL: {model} not supported"

if __name__ == "__main__":
    main()
