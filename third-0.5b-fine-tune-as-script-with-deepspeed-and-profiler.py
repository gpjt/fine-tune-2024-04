from datasets import load_dataset
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer

dataset_source = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_source)

base_model = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model).cuda()

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

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized


tokenized_dataset = dataset.map(tokenize_function, batched=True)

trainer = Trainer(
    model, args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
)


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        for step, inputs in enumerate(trainer.get_train_dataloader()):
            if step > 30:  # Profile for the first 30 steps
                break
            inputs = {k: v.cuda() for k, v in inputs.items()}  # Ensure inputs are on GPU
            loss = trainer.training_step(model, inputs)
            loss.backward()

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
