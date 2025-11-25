import os
import re
import time
import pandas as pd
from datasets import load_dataset
from openai import OpenAI


dataset = load_dataset("glue", "sst2", split="train[:2000]")
print("Total dataset size:", len(dataset))

client = OpenAI(api_key="sk-Iis58ApMv5b0BCmUDVMRPoPKL1N5YPmhoJoe2n1kESpfWfAM")

model = "gpt-3.5-turbo"

def call_openai(prompt, sentence, model=model):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": sentence}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

def extract_sentiment(text):
    match = re.findall(r"\b[01]\b", text)
    if match:
        return int(match[-1])
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

prompt = """Below is an instruction that describes a task, paired with 
an input that provides further context. Write a response that appropriately 
completes the request. 

Before answering, think carefully about the question and create a step-by-step
chain of thoughts to ensure a logical and accurate response. 

Analyze the given text from an online review and determine the sentiment 
polarity. Return a single number of either 0 and 1, with 0 being negative 
and 1 being the positive sentiment.  """

# ---------- 结果保存 ----------
result_dir = "./results_instruction_following"
os.makedirs(result_dir, exist_ok=True)

dataset_sizes = [16, 64, 256, 1024]

for size in dataset_sizes:
    print(f"\n===== Evaluating with {size} samples =====")
    records = []
    correct_count = 0
    turn_start_time = time.time()

    for i in range(size):
        start_time = time.time()
        item = dataset[i]
        sentence = item["sentence"]
        true_label = item["label"]

        model_response = call_openai(prompt, sentence, model)
        sentiment = extract_sentiment(model_response)

        if sentiment == true_label:
            correct_count += 1

        records.append({
            "id": i,
            "sentiment": sentiment,
            "true_label": true_label,
            "accuracy": None
        })

        elapsed = time.time() - start_time
        print(f"\nSentence {i}: {sentence}")
        print("Sentiment:", sentiment, " True_label:", true_label, f"Take time: {elapsed:.2f}s")

    turn_elapsed = time.time() - turn_start_time
    print(f"Turn time for {size} sentences: {turn_elapsed:.2f}s")

    accuracy = correct_count / size
    print(f"Accuracy for {size} sentences: {accuracy:.5f}")
    records[0]["accuracy"] = accuracy

    result_csv = pd.DataFrame(records)
    csv_path = os.path.join(result_dir, f"imdb_openai_gpt_3.5_turbo_{size}.csv")
    result_csv.to_csv(csv_path, index=False)
    print("Result saved to:", csv_path)
