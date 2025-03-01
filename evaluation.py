from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.api.registry import MODEL_REGISTRY
from peft import  PeftModel
import torch
from functools import partial

"""
TO BE ABLE TO USE THIS
(first try to use as is, if it fails try the following):

$ git clone https://github.com/EleutherAI/lm-evaluation-harness 
$ cd lm-evaluation-harness
$ git reset --hard 4d7d2f64576205105318fd12a622b6f0b7c70464
$ pip install -e .
"""

TASKS = [
    "winogrande",
    "piqa",
    "hellaswag",
]

@torch.no_grad()
def evaluate_on_nlp_tasks(
    model,
    tokenizer,
    max_length = 1024, 
    batch_size = 1,
    tasks = None, 
    limit = None, 
    return_samples = False,
    bootstrap_iters = 0,
    verbosity = "ERROR",
    do_shuffle = False,
    use_training = False,
    few_shot = None,
    make_valid_only = False,
):
    was_training = model.training
    lm_model = MODEL_REGISTRY["hf"](
        model,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        max_batch_size=batch_size,
        trust_remode_code=True,
    )

    tasks = tasks if tasks is not None else TASKS

    res = evaluator.simple_evaluate(
        model=lm_model,
        tasks = tasks,
        limit=limit,
        log_samples=return_samples,
        bootstrap_iters=bootstrap_iters,
        task_manager=TaskManager(),
        verbosity=verbosity,
        num_fewshot=few_shot,
    )
    model.train(was_training)
    return res

def evaluate_checkpoint(base_model, checkpoint, tokenizer, tasks=None, limit=None):
    model = PeftModel.from_pretrained(base_model, checkpoint)
    return evaluate_on_nlp_tasks(model, tokenizer, tasks=tasks, limit=limit)
