from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer

dataset_source = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_source)

base_model = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

model.to("cuda")

input("Press return to exit")
