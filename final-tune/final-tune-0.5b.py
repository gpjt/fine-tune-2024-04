from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer


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

    args = TrainingArguments(
        'outputs',
        learning_rate=8e-5,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        fp16=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        deepspeed="ds_config.json",
        report_to='none',
    )

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(tokenizer, examples),
        batched=True
    )

    trainer = Trainer(
        model, args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
    )

    trainer.train()

    model.push_to_hub("gpjt/Qwen1.5-0.5B-openassistant-guanaco-llama2-format")


if __name__ == "__main__":
    main(2)
