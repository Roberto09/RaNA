from trl import SFTTrainer
from transformers.data.data_collator import DataCollatorMixin, DataCollatorForLanguageModeling
from peft import LoraConfig

class SFTTrainer_(SFTTrainer):
    def _prepare_dataset(
        self,
        dataset,
        *args,
        **kwargs
    ):
        if isinstance(dataset, dict):
            return {k: self._prepare_dataset(v, *args, **kwargs) for k, v in dataset.items()}
        else:
            return super()._prepare_dataset(dataset, *args, **kwargs)

class QADataCollator(DataCollatorMixin):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.return_tensors = "pt"
        self.lm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def fill(self, tens, mx_len, val):
        extras = mx_len - len(tens)
        extras = torch.ones(extras, dtype=tens.dtype) * val
        return torch.concat([tens, extras])
    
    def torch_call(self, batch):
        if "labels" not in batch[0]:
            return self.lm_data_collator.torch_call(batch)
        inp_ids = [torch.from_numpy(b["input_ids"]) for b in batch]
        labels = [torch.from_numpy(b["labels"]) for b in batch]
        mask = [torch.from_numpy(b["attention_mask"]) for b in batch]
        mx_len = max(map(len, inp_ids))
        
        inp_ids = [self.fill(x, mx_len, self.tokenizer.eos_token_id) for x in inp_ids]
        labels = [self.fill(x, mx_len, -100) for x in labels]
        mask = [self.fill(x, mx_len, 1) for x in mask]
        batch = {
            "input_ids":torch.stack(inp_ids),
            "labels":torch.stack(labels),
            "attention_mask":torch.stack(mask)
        }
        return batch

def get_lora_config_llama(r=128, bias="none"):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=[
            'qkv.A',
            'qkv.B',
            'o_proj',
            'up_proj.A',
            'up_proj.B',
            'gate_proj.A',
            'gate_proj.B',
            'down_proj',
            'lm_head',
        ],
        modules_to_save=[
            'rotary_emb',
            'act'
            'input_layernorm',
            'post_attention_layernorm'
            'norm'
        ],
        bias=bias,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    return lora_config

def get_lora_config_gemma(r=128, bias="none"):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'up_proj.A',
            'up_proj.B',
            'gate_proj.A',
            'gate_proj.B',
            'down_proj',
            'lm_head',
        ],
        modules_to_save=[
            'rotary_emb',
            'act'
            'input_layernorm',
            'post_attention_layernorm'
            'norm'
        ],
        bias=bias,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    return lora_config