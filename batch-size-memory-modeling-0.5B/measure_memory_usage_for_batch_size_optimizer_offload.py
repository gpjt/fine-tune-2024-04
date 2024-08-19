import sys
import time

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer


class InterruptTraining(Exception):
    pass


class InterruptableTrainer(Trainer):
    END_ON_ITERATION = 30

    def training_step(self, model, inputs):
        step = self.state.global_step
        if step == 2:
            self.start_time = time.time()
        if step == self.END_ON_ITERATION:
            self.end_time = time.time()
            raise InterruptTraining()
        return super().training_step(model, inputs)

    def average_iterations_per_second(self):
        run_time = self.end_time - self.start_time
        return (self.END_ON_ITERATION - 1) / run_time


def tokenize_function(tokenizer, examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized


def main(batch_size):
    dataset_source = "timdettmers/openassistant-guanaco"
    dataset = load_dataset(dataset_source)

    base_model = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.gradient_checkpointing_enable()

    args = TrainingArguments(
        'outputs',
        learning_rate=8e-5,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        fp16=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        weight_decay=0.01,
        deepspeed="ds_config_optimizer_offload.json",
        report_to='none',
    )

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(tokenizer, examples),
        batched=True
    )

    trainer = InterruptableTrainer(
        model, args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
    )

    try:
        trainer.train()
    except InterruptTraining:
        pass
    except torch.cuda.OutOfMemoryError:
        with open("./results.csv", "a") as f:
            f.write(f"{batch_size}, OOM\n")
            return

    stats = torch.cuda.memory_stats()
    active_peak_mib = int(stats["active_bytes.all.peak"] / (1024 * 1024))
    reserved_peak_mib = int(stats["reserved_bytes.all.peak"] / (1024 * 1024))
    with open("./results.csv", "a") as f:
        f.write(f"{batch_size}, {active_peak_mib}, {reserved_peak_mib}, {trainer.average_iterations_per_second()}\n")


if __name__ == "__main__":
    main(int(sys.argv[2]))
