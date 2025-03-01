import os
import argparse
import pandas as pd
import torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from model_specific_utils import replace_qkv_layers_llama
from model_specific_utils import get_mlps_llama
import types
from evaluation import evaluate_on_nlp_tasks

MODEL_REVISIONS = {
    "meta-llama/Llama-2-7b-hf": "01c7f73d771dfac7d292323805ebc428287df4f9",
    "google/gemma-2b": "68e273d91b1d6ea57c9e6024c4f887832f7b43fa",
}

def lm_eval_rana(MODEL, DEVICE, DATASET_FILENAME, LOAD_ADAPTERS_FILENAME, LOAD_FT_STATE_DICT_FILENAME, PRUNE_RATIO):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{DEVICE}"

    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISIONS[MODEL])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISIONS[MODEL])
    
    is_llama = MODEL == "meta-llama/Llama-2-7b-hf"
    if is_llama: tokenizer.pad_token = tokenizer.eos_token

    if is_llama:
        rana_qkv_adapters = pd.read_pickle(f"./rana_models/{LOAD_ADAPTERS_FILENAME}_{PRUNE_RATIO}_qkvs.pkl")
        replace_qkv_layers_llama(model)
        for layer, rana_qkv_adapter in zip(model.model.layers, rana_qkv_adapters):
            layer.self_attn.q_proj.stacked_qkv.qkv = rana_qkv_adapter

    rana_adapters = pd.read_pickle(f"./rana_models/{LOAD_ADAPTERS_FILENAME}_{PRUNE_RATIO}_mlps.pkl")
    for layer, rana_mlp in zip(model.model.layers, rana_adapters):
        rana_mlp[0].down_proj = layer.mlp.down_proj
        layer.mlp = rana_mlp[0]

    model.load_state_dict(torch.load(f"./rana_models/{LOAD_FT_STATE_DICT_FILENAME}_{PRUNE_RATIO}_state_dict_finetuned.pkl"))
    model = model.eval()
    model = model.cuda()
    
    mlps = get_mlps_llama(model)

    def _efficient_get_activation_importances(self, fc1_o):
        with torch.no_grad():
            fc2_norm = self.down_proj_norm
            orig_shape = fc1_o.shape
            fc1_o = fc1_o.view(-1, orig_shape[-1]).contiguous()
            norms = fc1_o.abs() * fc2_norm
            norms = norms.view(orig_shape).contiguous()
            return norms.detach()
        
    def _efficient_forward(self, x):
        orig_fc1_out = self.compute_orig_fc1_out(x)
        with torch.no_grad():
            keep_mask = (self._get_activation_importances(orig_fc1_out) >= self.thresh).to(x)
            self.maybe_update_effective_rank(keep_mask)
        fc1_out = orig_fc1_out * keep_mask
        fc2_out = self.down_proj(fc1_out)
        return fc2_out
    
    for mlp in mlps:
        mlp.track_eff_rank = False
        mlp.up_proj.track_eff_rank = False
        mlp.activ.gate_proj.track_eff_rank = False
        with torch.no_grad():
            mlp.down_proj_norm = torch.linalg.norm(mlp.down_proj.weight, dim=0)
        mlp._get_activation_importances = types.MethodType(_efficient_get_activation_importances, mlp)
        mlp.forward = types.MethodType(_efficient_forward, mlp)

    tasks = [
        "winogrande",
        "piqa",
        "arc_easy",
        "hellaswag",
        "arc_challenge",
        "race",
    ]
    with torch.no_grad():
        if is_llama:
            eval_res = evaluate_on_nlp_tasks(model, tokenizer, few_shot=None, tasks=tasks, limit=10_000, max_length=2048, bootstrap_iters=1000, do_shuffle=True, use_training=False)
        else:
            eval_res = evaluate_on_nlp_tasks(model, tokenizer, few_shot=5, tasks=tasks, limit=10_000, bootstrap_iters=1000, do_shuffle=True, use_training=False)
    print(f"Benchmark Evals: {eval_res['results']}")

def main():
    parser = argparse.ArgumentParser(description="lm_eval_harness_rana")

    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", type=str, help="Model Name")
    parser.add_argument("--prune_ratio", type=float, help="Prune Ratio", required=True)
    parser.add_argument("--device", default=0, type=int, help="CUDA Device ID")
    parser.add_argument("--dataset_filename", default="red_pajama_dataset", type=str, help="dataset file name (where to save / where to load from the dataset")
    parser.add_argument("--load_rana_filename", default="rana_non_ft", type=str, help="filename from which rana adapters should be loaded")
    parser.add_argument("--load_rana_ft_sate_dict_filename", default="rana", type=str, help="filename from which rana-adapted fine-tuned model state dict should be loaded")

    args = parser.parse_args()

    model = args.model
    prune_ratio = args.prune_ratio
    device = args.device
    dataset_filename = args.dataset_filename
    load_rana_filename = args.load_rana_filename
    load_rana_ft_state_dict_filename = args.load_rana_ft_sate_dict_filename
    
    lm_eval_rana(model, device, dataset_filename, load_rana_filename, load_rana_ft_state_dict_filename, prune_ratio)

    assert model in MODEL_REVISIONS, f"MODEL: {model} not supported"

if __name__ == "__main__":
    main()