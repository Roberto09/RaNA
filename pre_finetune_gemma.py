import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from dataset import get_baseline_dataset_v2
from post_training import *
from trainers import SFTTrainer_, QADataCollator
from peft import LoraConfig
from model_specific_utils import *


MODEL_REVISIONS = {
    "meta-llama/Llama-2-7b-hf": "01c7f73d771dfac7d292323805ebc428287df4f9",
    "google/gemma-2b": "68e273d91b1d6ea57c9e6024c4f887832f7b43fa",
}

def get_lora_config(r=16, bias="none"):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=[
            'gate_proj', # re-train prunned layers for now
            'up_proj',
            'down_proj',
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj'
        ],
        bias=bias,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    return lora_config

def pre_finetune_gemma(LR, TSTEPS, MBS, DEVICE, DATASET_FILENAME, SAVE_FILENAME):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{DEVICE}"
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", revision=MODEL_REVISIONS["google/gemma-2b"])
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", revision=MODEL_REVISIONS["google/gemma-2b"])
    
    dataset = get_baseline_dataset_v2("togethercomputer/RedPajama-Data-1T-Sample", f"{DATASET_FILENAME}.pkl", train_size=100_000)
    train_data, eval_data = dataset["train"], dataset["test"]

    training_arguments = get_training_arguments("./tmp", micro_batch_size=MBS, logging_steps=10, linear_lr_decay=False, eval_steps=100, epochs=1, weight_decay=0, learning_rate=LR)
    lora_config = get_lora_config()

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
        max_seq_length=2048,
        callbacks=callbacks,
        data_collator=QADataCollator(tokenizer),
    )
    
    eval_res_pretrain = trainer.evaluate()["eval_red_pajama_loss"]
    print(f"Eval-Red-Pajama loss Pre-finetune: {eval_res_pretrain}")
    trainer.train()

    eval_res_posttrain = trainer.evaluate()["eval_red_pajama_loss"]
    print(f"Eval-Red-Pajama loss Post-finetune: {eval_res_posttrain}")
    
    lora_model = trainer.model.base_model
    lora_model.merge_and_unload()

    torch.save(model.state_dict(), f"rana_models/{SAVE_FILENAME}.pkl")

def main():
    parser = argparse.ArgumentParser(description="pre_fine_tune_gemma")

    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning Rate")
    parser.add_argument("--train_steps", default=700, type=int, help="Train Steps")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch Size")
    parser.add_argument("--device", default=0, type=int, help="CUDA Device ID")
    parser.add_argument("--dataset_filename", default="red_pajama_dataset", type=str, help="dataset file name (where to save / where to load from the dataset")
    parser.add_argument("--save_filename", default="gemma_red_pajama_state_dict", type=str, help="filename where pre-fine-tuned gemma state dict should be saved")

    args = parser.parse_args()

    learning_rate = args.learning_rate
    train_steps = args.train_steps
    batch_size = args.batch_size
    device = args.device
    dataset_filename = args.dataset_filename
    save_filename = args.save_filename
    
    pre_finetune_gemma(learning_rate, train_steps, batch_size, device, dataset_filename, save_filename)

if __name__ == "__main__":
    main()



