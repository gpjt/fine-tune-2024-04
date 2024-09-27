from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments


def tokenize_function(tokenizer, examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized


def main(batch_size):
    dataset_source = "gpjt/openassistant-guanaco-llama2-format"
    dataset = load_dataset(dataset_source)

    base_model = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)

    args = TrainingArguments(
        'outputs',
        learning_rate=8e-5,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=9,
        weight_decay=0.01,
        deepspeed="ds_config.json",
        report_to='none',
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        push_to_hub=False,
    )
    early_stopping = EarlyStoppingCallback(early_stopping_patience=1)

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(tokenizer, examples),
        batched=True
    )

    trainer = Trainer(
        model, args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        callbacks=[early_stopping],
    )

    trainer.train()

    trainer.save_model("final-result")


if __name__ == "__main__":
    main(2)
