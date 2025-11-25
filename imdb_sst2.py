import pandas as pd

from datasets import load_dataset

dataset = load_dataset("glue", "sst2", split="train")
test = load_dataset("glue", "sst2", split="test")

cnt = 0
for i in range(len(test)):
    if test[i]["label"] == -1:
        cnt += 1
print(cnt/len(test))