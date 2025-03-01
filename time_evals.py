import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import get_baseline_dataset_v2
from transformers import DataCollatorForLanguageModeling

def get_dataset(path, tokenizer, max_seq_len):
    dataset = get_baseline_dataset_v2("togethercomputer/RedPajama-Data-1T-Sample", path, train_size=100_000)

    dataset = dataset["train"]
    dataset = dataset.select(range(1000))
    def tokenize(entry):
        res = tokenizer(entry["text"],
            truncation=True,
            max_length=max_seq_len,
            return_overflowing_tokens=True,
        )
        return {
            "input_ids": res["input_ids"][0],
            "attention_mask": res["attention_mask"][0],
        } 
    dataset = dataset.map(tokenize, remove_columns=["text", "meta"])
    return dataset

# Adapted from: https://github.com/ScalingIntelligence/CATS.
def benchmark_decode(model, tokenizer, dataset_path, B=1, bm=1, gen_len=501, iters=50):
    save_dir = model.save_dir
    save_path = os.path.join(save_dir, "throughput.csv")

    with open(save_path, "w") as f:
        print(
            "context_length,decoded_tokens,elapsed_time_ms,decoded_tokens_per_sec",
            file=f,
        )

    test_dataset = get_dataset(dataset_path, tokenizer, max_seq_len=1000)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    dataloader = DataLoader(test_dataset, batch_size=B, collate_fn=data_collator)
    model.eval()
    print(len(dataloader))
    count = 0
    max_count = iters
    with torch.no_grad():
        for batch in tqdm(dataloader, total=max_count):
            count += 1
            if count > max_count:
                break
            input_ids = batch["input_ids"].cuda()
            max_lengths = input_ids.size(1) + gen_len
            outputs = model.generate(
                input_ids=input_ids,
                max_length=max_lengths,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                num_beams=bm,
            )
