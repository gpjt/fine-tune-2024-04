import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def upload_model(local_model_name, remote_model_name):
    tokenizer = AutoTokenizer.from_pretrained(local_model_name)
    model = AutoModelForCausalLM.from_pretrained(local_model_name, torch_dtype=torch.bfloat16)

    tokenizer.push_to_hub(remote_model_name)
    model.push_to_hub(remote_model_name)




if __name__ == "__main__":
    upload_model(sys.argv[1], sys.argv[2])
