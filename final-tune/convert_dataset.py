import re

from datasets import load_dataset


dataset_source = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_source)


prompt_template = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{question} [/INST]
{response}
"""

pattern = r"### Human: (.*?)### Assistant: (.*)"

def rewrite_prompts(examples):
    responses = []
    # Iterate over each example
    for text in examples["text"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            question = match.group(1).strip()
            response = match.group(2).strip()
            reformatted_text = prompt_template.format(question=question, response=response)
            while "### Human: " in reformatted_text:
                reformatted_text = reformatted_text.replace("### Human: ", "[INST]", 1)
                if "### Assistant: " in reformatted_text:
                    reformatted_text = reformatted_text.replace("### Assistant: ", "[/INST]\n", 1)
                else:
                    reformatted_text += "[/INST]\n"

            responses.append(reformatted_text)
        else:
            # You might want to handle errors differently
            responses.append("Error: Did not match expected pattern.")
    return {"reformatted_text": responses}


reformatted_dataset = dataset.map(rewrite_prompts, batched=True)


for row in list(reformatted_dataset["train"]) + list(reformatted_dataset["test"]):
    if row["reformatted_text"] == "Error: Did not match expected pattern.":
        print(row["text"])


human_prompts = []
assistant_prompts = []
for row in list(reformatted_dataset["train"]) + list(reformatted_dataset["test"]):
    if "### Human: " in row["reformatted_text"]:
        human_prompts.append(row["reformatted_text"])
    if "### Assistant: " in row["reformatted_text"]:
        assistant_prompts.append(row["reformatted_text"])

assert len(human_prompts) == 0
assert len(assistant_prompts) == 0

reformatted_dataset.push_to_hub("gpjt/openassistant-guanaco-llama2-format")
