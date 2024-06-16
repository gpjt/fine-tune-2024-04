from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers.utils import is_sagemaker_mp_enabled

def print_memory_usage(step, stage):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"Step {step} ({stage}): Allocated: {allocated / (1024 ** 3):.2f} GB, Reserved: {reserved / (1024 ** 3):.2f} GB")
    print(torch.cuda.memory_summary())

class MemoryLoggingTrainer(Trainer):
    def training_step(self, model, inputs):
        step = self.state.global_step
        print_memory_usage(step, "before training_step")

        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            loss = loss_mb.reduce_mean().detach().to(self.args.device)
        else:
            with self.compute_loss_context_manager():
                print_memory_usage(step, "before forward pass")
                loss = self.compute_loss(model, inputs)
                print_memory_usage(step, "after forward pass")

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            print_memory_usage(step, "before backward pass")
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            print_memory_usage(step, "after backward pass")

        print_memory_usage(step, "after training_step")
        return loss.detach() / self.args.gradient_accumulation_steps

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

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2048)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized


tokenized_dataset = dataset.map(tokenize_function, batched=True)

trainer = MemoryLoggingTrainer(
    model, args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
)

trainer.train()
