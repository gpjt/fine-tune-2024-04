import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from prompt import prompt_template


def ask_question(model, tokenizer, question):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
    prompt = prompt_template.format(question=question, response="")
    tokens_in = len(tokenizer(prompt)["input_ids"])
    start = time.time()
    result = pipe(prompt)
    end = time.time()
    generated_text = result[0]['generated_text']
    tokens_out = len(tokenizer(generated_text)["input_ids"])
    print(generated_text)
    tokens_generated = tokens_out - tokens_in
    time_taken = end - start
    tokens_per_second = tokens_generated / time_taken
    print(f"{tokens_generated} tokens in {time_taken:.2f}s: {tokens_per_second:.2f} tokens/s)")



def test_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")

    question = input("You: ")
    ask_question(model, tokenizer, question)




if __name__ == "__main__":
    test_model(sys.argv[1])
