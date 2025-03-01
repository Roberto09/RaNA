import numpy as np
import torch
from tqdm import tqdm
from model_specific_utils import get_linears_llama
import gc

def get_tokenized_train_data(train_data, tokenizer, ttl_tokens, seq_size):
    np.random.seed(123)
    tokenized_train_data = []
    token_count = 0
    for tdata in tqdm(train_data.select(range(int((ttl_tokens // seq_size)*5)))):
        ttdata = tokenizer(tdata["text"])["input_ids"]
        if len(ttdata) < seq_size: continue
        
        start = np.random.randint(0, len(ttdata)-seq_size+1) 
        tokenized_train_data.append(ttdata[start:start+seq_size])
        token_count += seq_size
    return tokenized_train_data


def format_input_sample(inp, sample_size=32_000):
    inp = torch.concat(inp).view(-1, inp[0].shape[-1])
    rand_idxs = torch.randperm(inp.shape[0])[:sample_size]
    inp = inp[rand_idxs]
    return inp

def _get_input_tensors(model, tokenized_train_data, inp_sample_size=32_000, seq_size=1024, ttl_tokens_mult=2, get_linears=get_linears_llama, mem_efficient=False):
    inp_tensors = {}
    linear_layers = get_linears(model)
    for layers in linear_layers:
        for layer in layers: inp_tensors[layer] = []

    torch.manual_seed(42)
    ttl_batches = inp_sample_size*ttl_tokens_mult//seq_size
    def hook_fn(module, input, output):
        inp = input[0].view(-1, input[0].shape[-1]).detach().cpu()
        if mem_efficient:
            to_pick = (inp_sample_size + ttl_batches) // ttl_batches
            rand_idxs = torch.randperm(inp.shape[0])[:to_pick]
            inp = inp[rand_idxs]
        inp_tensors[module].append(inp)
    
    handles = []
    for module in inp_tensors.keys():
        handles.append(module.register_forward_hook(hook_fn))
    
    print("gathering input samples")
    model = model.cuda().eval()
    with torch.no_grad():
        for batch_i in tqdm(range(ttl_batches)):
            batch_start, batch_end = batch_i, batch_i+1
            examples = torch.stack([torch.tensor(d) for d in tokenized_train_data[batch_start:batch_end]]).cuda()
            model(examples, labels=examples)
    for handle in handles: handle.remove()
    torch.cuda.empty_cache()
    gc.collect()
    print("done gathering input samples")

    orig_inp_tens = inp_tensors
    inp_tensors = {}
    torch.manual_seed(42)
    with torch.no_grad():
        for layer_i, layers in tqdm(enumerate(linear_layers)):
            new_entries = []
            for layer in layers:
                new_entries.append(format_input_sample(orig_inp_tens[layer], sample_size=inp_sample_size).cpu())
            inp_tensors[layer_i] = new_entries
    
    return inp_tensors

def get_input_tensors(model, tokenizer, train_data, inp_sample_size=32_000, seq_size=1024, ttl_tokens_mult=2, get_linears=get_linears_llama, mem_efficient=False):
    print("getting tokenized data")
    tokenized_train_data = get_tokenized_train_data(train_data, tokenizer, ttl_tokens=inp_sample_size*ttl_tokens_mult, seq_size=seq_size)
    print("done getting tokenized data")
    return _get_input_tensors(model, tokenized_train_data, inp_sample_size, seq_size, ttl_tokens_mult, get_linears, mem_efficient) 