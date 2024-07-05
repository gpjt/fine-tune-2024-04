import sys

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer


class InterruptTraining(Exception):
    pass


class InterruptableTrainer(Trainer):
    def training_step(self, model, inputs):
        step = self.state.global_step
        if step == 30:
            raise InterruptTraining()
        return super().training_step(model, inputs)


def tokenize_function(tokenizer, sequence_length, examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=sequence_length)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized


def main(sequence_length):
    dataset_source = "timdettmers/openassistant-guanaco"
    dataset = load_dataset(dataset_source)

    base_model = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    batch_size = 1
    args = TrainingArguments(
        'outputs',
        learning_rate=8e-5,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        fp16=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=2,
        weight_decay=0.01,
        deepspeed="ds_config.json",
        report_to='none',
    )

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(tokenizer, sequence_length, examples),
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

    stats = torch.cuda.memory_stats()
    active_peak_mib = int(stats["active_bytes.all.peak"] / (1024 * 1024))
    reserved_peak_mib = int(stats["reserved_bytes.all.peak"] / (1024 * 1024))
    print(f"{sequence_length}, {active_peak_mib}, {reserved_peak_mib}")


if __name__ == "__main__":
    main(int(sys.argv[2]))
