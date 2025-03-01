import argparse
import os
import pandas as pd
import torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from dataset import get_baseline_dataset_v2
from model_specific_utils import replace_qkv_layers_llama
from post_training import *
from model_specific_utils import get_mlps_llama
from trainers import SFTTrainer_, QADataCollator, get_lora_config_llama, get_lora_config_gemma
from model_specific_utils import *
from evaluation import evaluate_on_nlp_tasks
import types
import gc

MODEL_REVISIONS = {
    "meta-llama/Llama-2-7b-hf": "01c7f73d771dfac7d292323805ebc428287df4f9",
    "google/gemma-2b": "68e273d91b1d6ea57c9e6024c4f887832f7b43fa",
}

def fine_tune_rana(MODEL, LR, TSTEPS, MBS, DEVICE, DATASET_FILENAME, LOAD_FILENAME, PRUNE_RATIO, SAVE_FILENAME, PRE_LOAD_MODEL_STATE_DICT):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{DEVICE}"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISIONS[MODEL])
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISIONS[MODEL])
    if PRE_LOAD_MODEL_STATE_DICT is not None:
        model.load_state_dict(torch.torch.load(PRE_LOAD_MODEL_STATE_DICT))
    
    dataset = get_baseline_dataset_v2("togethercomputer/RedPajama-Data-1T-Sample", f"{DATASET_FILENAME}.pkl", train_size=100_000)
    train_data, eval_data = dataset["train"], dataset["test"]

    is_llama = MODEL == "meta-llama/Llama-2-7b-hf"
    rana_adapters = pd.read_pickle(f"./rana_models/{LOAD_FILENAME}_{PRUNE_RATIO}_mlps.pkl")
    if is_llama:
        rana_qkv_adapters = pd.read_pickle(f"./rana_models/{LOAD_FILENAME}_{PRUNE_RATIO}_qkvs.pkl")
        gc.collect(); torch.cuda.empty_cache()

        replace_qkv_layers_llama(model)
        for layer, rana_qkv_adapter in zip(model.model.layers, rana_qkv_adapters):
            layer.self_attn.q_proj.stacked_qkv.qkv = rana_qkv_adapter

    gc.collect(); torch.cuda.empty_cache()
    for layer, rana_mlp in zip(model.model.layers, rana_adapters):
        rana_mlp[0].down_proj = layer.mlp.down_proj
        layer.mlp = rana_mlp[0]

    training_arguments = get_training_arguments("./tmp", micro_batch_size=MBS, logging_steps=10, linear_lr_decay=False, eval_steps=100, epochs=1, weight_decay=0, learning_rate=LR)
    lora_config = get_lora_config_llama(r=64) if is_llama else get_lora_config_gemma()

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    train_data, eval_data = dataset["train"], dataset["test"]
    eval_datasets = {
        "red_pajama":eval_data.select(np.arange(300)),
    }

    tsteps = TSTEPS
    train_data = train_data.select(np.arange(60*tsteps))
    callbacks = []

    if is_llama: tokenizer.pad_token = tokenizer.eos_token
    training_arguments.save_strategy="no"
    training_arguments.per_device_eval_batch_size = 1
    
    model.enable_input_require_grads()

    trainer = SFTTrainer_(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_datasets,
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        dataset_text_field="text",
        max_seq_length=1024,
        callbacks=callbacks,
        data_collator=QADataCollator(tokenizer),
    )
    
    eval_res_pretrain = trainer.evaluate()["eval_red_pajama_loss"]
    print(f"Eval-Red-Pajama loss Pre-finetune: {eval_res_pretrain}")
    trainer.train()

    eval_res_posttrain = trainer.evaluate()["eval_red_pajama_loss"]
    
    if is_llama:
        qkvs = [layer.self_attn.q_proj.stacked_qkv.qkv for layer in model.model.layers]
        for qkv in qkvs:
            qkv.running_effective_rank.reset()
    
    mlps = get_mlps_llama(model)
    for mlp in mlps:
        mlp.up_proj.running_effective_rank.reset()
        mlp.activ.gate_proj.running_effective_rank.reset()
        mlp.running_effective_rank.reset()
    
    trainer.evaluate()

    if is_llama :
        fps_ratio = []
        for qkv in qkvs:
            fps_ratio.append(qkv.get_effective_flops_ratio().item())
        fps_ratio_qkvs = np.mean(fps_ratio)
    
    mlps = get_mlps_llama(model)
    fps_ratio = []
    for mlp in mlps:
        fps_ratio.append(mlp.get_effective_flops_ratio().item())
    fps_ratio_mlps = np.mean(fps_ratio)

    print(f"Eval-Red-Pajama loss Post-finetune: {eval_res_posttrain}")
    print("Effective FLOPs ratios post-finetune:")
    if is_llama: print(f"Percent of used flops QKVs: {fps_ratio_qkvs*100}%, MLPs: {fps_ratio_mlps*100}%")
    else: print(f"Percent of used flops MLPs: {fps_ratio_mlps*100}%")
    
    lora_model = trainer.model.base_model
    lora_model.merge_and_unload()

    mlps = get_mlps_llama(model)

    torch.save(model.state_dict(), f"rana_models/{SAVE_FILENAME}_{PRUNE_RATIO}_state_dict_finetuned.pkl")

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

def get_default_lr(prune_ratio, model):
    if model == "meta-llama/Llama-2-7b-hf":
        if prune_ratio <= 0.3: return 5e-5
        elif prune_ratio <= 0.4: return 1e-5
        elif prune_ratio <= 0.46: return 2.5e-4
        elif prune_ratio <= 0.53: return 5e-5
        else: return 2.5e-4
    else:
        return 2.5e-4

def main():
    parser = argparse.ArgumentParser(description="fine_tune_rana_model")

    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", type=str, help="Model Name")
    parser.add_argument("--prune_ratio", type=float, help="Prune Ratio", required=True)
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning Rate, if not specified uses default")
    parser.add_argument("--train_steps", default=500, type=int, help="Train Steps")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch Size")
    parser.add_argument("--device", default=0, type=int, help="CUDA Device ID")
    parser.add_argument("--dataset_filename", default="red_pajama_dataset", type=str, help="dataset file name (where to save / where to load from the dataset")
    parser.add_argument("--load_rana_filename", default="rana_non_ft", type=str, help="filename from which rana adapters should be loaded")
    parser.add_argument("--save_filename", default="rana", type=str, help="filename where fine-tuned rana state dict should be saved")
    parser.add_argument("--pre-load-model-state-dict", default=None, type=str, help="State-dict to load into model before generating rana adapters")

    args = parser.parse_args()

    model = args.model
    assert model in MODEL_REVISIONS, f"MODEL: {model} not supported"
    
    prune_ratio = args.prune_ratio
    learning_rate = args.learning_rate if args.learning_rate is not None else get_default_lr(prune_ratio, model)
    train_steps = args.train_steps
    batch_size = args.batch_size
    device = args.device
    dataset_filename = args.dataset_filename
    load_rana_filename = args.load_rana_filename
    save_filename = args.save_filename
    pre_load_model_state_dict = args.pre_load_model_state_dict
    

    fine_tune_rana(model, learning_rate, train_steps, batch_size, device, dataset_filename, load_rana_filename, prune_ratio, save_filename, pre_load_model_state_dict)

if __name__ == "__main__":
    main()
