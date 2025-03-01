import pickle
import torch
from torch import nn

def get_linears_llama(model):
    linear_layers = []
    for layer in model.model.layers:
        mlp = layer.mlp
        linear_layers.append([mlp.up_proj, mlp.down_proj, mlp.gate_proj])
    return linear_layers

def get_mlps_llama(model):
    return [layer.mlp for layer in model.model.layers]

def get_estimated_pkl_bytes(data):
    serialized_data = pickle.dumps(data)
    size_of_pickle = len(serialized_data)
    print(f"Estimated size of the pickle file: {size_of_pickle} bytes")
    return size_of_pickle

class StackedQKV(nn.Module):
    def __init__(self, q_layer, k_layer, v_layer):
        super().__init__()
        assert q_layer.bias == k_layer.bias == v_layer.bias == None
        self.out_dim, self.inp_dim = q_layer.weight.shape
        self.qkv = nn.Linear(in_features=self.inp_dim, out_features=self.out_dim*3, bias=False)
        self.qkv.weight.data = torch.concat([q_layer.weight.detach(), k_layer.weight.detach(), v_layer.weight.detach()])
        self.curr_out = None
    
    def forward(self, x, layer_type):
        if layer_type == "query":
            self.curr_out = self.qkv(x)
        
        if layer_type == "query":
            return self.curr_out[:, :, 0:self.out_dim]
        if layer_type == "key":
            return self.curr_out[:, :, self.out_dim:self.out_dim*2]
        if layer_type == "value":
            return self.curr_out[:, :, self.out_dim*2:self.out_dim*3]
        assert False

class CustomQKVProj(nn.Module):
    def __init__(self, stacked_qkv, layer_type):
        super().__init__()
        self.stacked_qkv = stacked_qkv
        self.layer_type = layer_type
    def forward(self, x):
        return self.stacked_qkv(x, self.layer_type)

def replace_qkv_layers_llama(model):
    for layer in model.model.layers:
        attn = layer.self_attn
        stacked_qkv = StackedQKV(attn.q_proj, attn.k_proj, attn.v_proj)
        attn.q_proj = CustomQKVProj(stacked_qkv, "query")
        attn.k_proj = CustomQKVProj(stacked_qkv, "key")
        attn.v_proj = CustomQKVProj(stacked_qkv, "value")