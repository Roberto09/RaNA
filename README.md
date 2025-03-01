# Adaptive Rank Allocation
This is the implementation for the Rank and Neuron Adapters (RaNA) adapters from **Adaptive Rank Allocation: Speeding Up Modern Transformers With RaNA Adapters**.

## Running Rana
### Requirements:
First you will have to install requirements and [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). To do that, on a clean conda environment, run:
```
$ sh setup.sh
```

### Generating RaNA adapters
To generate the rana adapters, on llama-2-7b, run the following script, where `prune_ratio` is the target prune ratio for the rana-adapted layers. This will generate two files inside the `./rana_models`, which contain the MLP and QKV RaNA adapters.
```
$ python generate_rana_model.py  \
--model meta-llama/Llama-2-7b-hf \
--prune_ratio <prune_ratio> \
--device 0 \
--dataset_filename red_pajama_dataset \
--save_filename rana_non_ft_adapters 2>&1 | tee generate.out
```

### Fine-Tuning RaNA adapted Model
To fine-tune the RaNA adapted model, run the following script. This will generate a new file inside the `./rana_models` directory, containing the state dict for the RaNA adapted model.
```
$ python fine_tune_rana_model.py  \
--model meta-llama/Llama-2-7b-hf \
--prune_ratio <prune_ratio> \
--device 0 \
--dataset_filename red_pajama_dataset \
--load_rana_filename rana_non_ft_adapters \
--save_filename rana 2>&1 | tee fine_tune.out
```

### Evaluating RaNA adapted Model w/ LM Eval Harness
To evaluate the RaNA adapted model on LM-eval-harness benchmarks, run the following script.
```
$ python lm_eval_harness_rana.py \
--model meta-llama/Llama-2-7b-hf \
--prune_ratio <prune_ratio> \
--device 0 \
--dataset_filename red_pajama_dataset \
--load_rana_filename rana_non_ft_adapters \
--load_rana_ft_sate_dict_filename rana 
```

### Assessing latency of RaNA adapted Model
To assess the latency of the RaNA adapted model, run the following script. This will generate directory `./latency_rana_results...` with a csv file inside with the gathered latency metrics.
```
$ python latency_eval_rana.py \
--model meta-llama/Llama-2-7b-hf \
--prune_ratio <prune_ratio> \
--device 0 \
--dataset_filename red_pajama_dataset \
--load_rana_filename rana_non_ft_adapters \
--load_rana_ft_sate_dict_filename rana \
--lat_save_dir latency_rana_results
```

### Pre-Fine-Tuning Gemma
To pre-fine tune Gemma-2b on the RedPajama dataset (before generating adapters and fine-tuning them), run the following script. This will generate a state_dict `gemma_red_pajama_state_dict` inside `./rana_models`.
```
$ python pre_finetune_gemma.py --device 0
```
Further, you can generate_adapters and fine_tune adapters by adding/substituting the following flags in the previous examples:
```
--model google/gemma-2b \
--pre-load-model-state-dict /root/rana-os-w-gemma/rana_models/gemma_red_pajama_state_dict.pkl \
```