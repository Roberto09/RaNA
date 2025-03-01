import transformers
from transformers import SchedulerType

def get_training_arguments(output_dir, micro_batch_size=6, logging_steps=10, linear_lr_decay=True, learning_rate=1e-4,
                           eval_steps=200, weight_decay=0.0, epochs=2, fp16=False, load_best_model_at_end=True):
    batch_size = 60
    gradient_accumulation_steps = batch_size // micro_batch_size

    if linear_lr_decay:
        scheduler_args = dict(warmup_steps=100)
    else:
        scheduler_args = dict(lr_scheduler_type=SchedulerType.CONSTANT)

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=fp16,
        logging_steps=logging_steps,
        logging_first_step=True,
        optim="adamw_torch",
        weight_decay=weight_decay,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=20,
        load_best_model_at_end=load_best_model_at_end,
        ddp_find_unused_parameters=None,
        group_by_length=False,
        report_to=[],
        **scheduler_args,
    )
    return training_arguments

