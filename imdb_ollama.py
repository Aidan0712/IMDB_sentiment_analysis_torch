import ollama
from datasets import load_dataset

import re
import os
import pandas as pd
import time

dataset = load_dataset("glue", "sst2", split="train[:2000]")
print("Total dataset size:", len(dataset))

def call_ollama(prompt, model):
    response = ollama.generate(
        model=model,
        prompt=prompt
    )
    return response["response"]

def extract_sentiment(text):
    match = re.findall(r"\b[01]\b", text)

    if match:
        return int(match[-1])
    else:
        return None

prompt_style = """Below is an instruction that describes a task, paired with 
an input that provides further context. Write a response that appropriately 
completes the request. 

Before answering, think carefully about the question and create a step-by-step
chain of thoughts to ensure a logical and accurate response. 

### Instruction: 
Analyze the given text from an online review and determine the sentiment 
polarity. Return a single number of either 0 and 1, with 0 being negative 
and 1 being the positive sentiment.  

### Input: 
{} 

### Response:
<think>
"""

model_result_map = {
    "deepseek-r1:8b":  "ds_r1_8b",
    "qwen3:8b":  "qw3_8b",
    "llama3.1:8b": "lm3.1_8b",
    "gemma2:9b": "gemma2_9b",
    "phi4-mini:3.8b": "phi4_mini_3.8b",
    "mistral:7b": "mistral_7b",
}

model = "mistral:7b"

records = []
correct_count = 0

result_dir = "./results_instruction_following"
os.makedirs(result_dir, exist_ok=True)  # 如果文件夹不存在会自动创建

save_sizes = {16, 64, 256, 512, 1024}

max_size = max(save_sizes)

turn_time = time.time()

for i in range(16):
    start_time = time.time()
    item = dataset[i]
    sentence = item["sentence"]
    true_label = item["label"]

    prompt = prompt_style.format(sentence)
    model_response = call_ollama(prompt, model)
    sentiment = extract_sentiment(model_response)

    if sentiment == true_label:
        correct_count += 1

    records.append({
        "id": i,
        "sentiment": sentiment,
        "true_label": true_label,
        "accuracy": None
    })

    take_time = time.time() - start_time

    print(f"\nSentence {i+1} :", sentence)
    print("Sentiment:", sentiment, " True_label:", true_label, " Take time:", take_time)

    curr = i + 1
    if curr in save_sizes:
        turn_time_used = time.time() - turn_time
        print(f"Turn time for {curr} sentences:", turn_time_used)

        accuracy = correct_count / curr
        print(f"Accuracy for {curr} sentences: {accuracy:.5f}")

        records[0]["accuracy"] = accuracy

        result_csv = pd.DataFrame(records)
        model_name = model_result_map.get(model, 'default')
        csv_path = os.path.join(result_dir, f"imdb_ollama_{model_name}_{curr}.csv")
        result_csv.to_csv(csv_path, index=False)
        print("Result saved to:", csv_path)