import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from prompt import prompt_template


def ask_question(model, tokenizer, prompt, debug):
    if debug:
        print(">>> sending to model")
        print(prompt)
        print("<<< end of send")
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
    start = time.time()
    result = pipe(prompt, return_full_text=False)
    end = time.time()
    generated_text = result[0]['generated_text']
    tokens_out = len(tokenizer(generated_text)["input_ids"])
    if debug:
        print(">>> model returns")
        print(generated_text)
        print("<<< end model result")
    time_taken = end - start
    tokens_per_second = tokens_out / time_taken
    model_response_clipped = generated_text.split("[INST]")[0]
    return model_response_clipped, tokens_out, tokens_per_second, time_taken



def test_model(model_name, debug):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16)

    question = input("You: ")
    prompt = prompt_template.format(question=question, response="")

    while True:
        response, tokens_generated, tokens_per_second, time_taken = ask_question(model, tokenizer, prompt, debug)
        print("Bot: ", response)
        print(f"{tokens_generated} tokens in {time_taken:.2f}s: {tokens_per_second:.2f} tokens/s)")
        prompt += response
        question = input("You: ")
        prompt += f"[INST]\n{question} [/INST]"


if __name__ == "__main__":
    if "--debug" in sys.argv:
        debug = True
        sys.argv.remove("--debug")
    else:
        debug = False

    if len(sys.argv) < 2:
        print("Usage: python script.py [--debug] <model_name>")
        sys.exit(1)

    test_model(sys.argv[1], debug)
