import os
import pandas as pd

# 当前目录路径
folder = "./result"

# 存储结果
results = []

def rate_to_sentiment(r):
    if r < 5:
        return 0
    elif r >= 7:
        return 1
    else:
        return -1

# 遍历目录下所有csv文件
for file in os.listdir(folder):
    if file.endswith(".csv") and not file.startswith("accuracy_summary"):
        path = os.path.join(folder, file)
        print(f"正在处理：{file}")

        # 读取csv
        df = pd.read_csv(path)

        # 提取评分
        df["rating"] = df["id"].apply(lambda x: int(x.split("_")[1]))

        # 根据评分生成真实情绪标签
        df["true_sentiment"] = df["rating"].apply(rate_to_sentiment)

        # 判断是否预测正确
        df["correct"] = (df["true_sentiment"] == df["sentiment"]).astype(int)

        # 计算准确率
        acc = df["correct"].mean()
        print(f"{file} 的准确率: {acc * 100:.2f}%")

        results.append({"file": file, "accuracy": acc})



# 汇总结果为一个DataFrame
summary = pd.DataFrame(results)
summary.sort_values(by="accuracy", ascending=False, inplace=True)

# 保存汇总结果
summary.to_csv(os.path.join(folder, "accuracy_summary.csv"), index=False, encoding="utf-8-sig")

print("\n所有文件处理完成，汇总结果已保存为 accuracy_summary.csv")
print(summary)


results = []


folder2 = "./results_instruction_following"

for file in os.listdir(folder2):
    if file.endswith(".csv") and not file.startswith("accuracy_summary"):
        path = os.path.join(folder2, file)
        print(f"正在处理：{file}")

        # 读取csv
        df = pd.read_csv(path)

        # 判断是否预测正确
        df["correct"] = (df["true_label"] == df["sentiment"]).astype(int)

        # 计算准确率
        acc = df["correct"].mean()
        print(f"{file} 的准确率: {acc * 100:.2f}%")

        results.append({"file": file, "accuracy": acc})

# 汇总结果为一个DataFrame
summary = pd.DataFrame(results)
summary.sort_values(by="accuracy", ascending=False, inplace=True)

# 保存汇总结果
summary.to_csv(os.path.join(folder2, "accuracy_summary.csv"), index=False, encoding="utf-8-sig")

print("\n所有文件处理完成，汇总结果已保存为 accuracy_summary.csv")
print(summary)

results = []

folder3 = "./results_instruction_tuning"

for file in os.listdir(folder3):
    if file.endswith(".csv") and not file.startswith("accuracy_summary"):
        path = os.path.join(folder3, file)
        print(f"正在处理：{file}")

        # 读取csv
        df = pd.read_csv(path)

        # 提取评分
        df["rating"] = df["id"].apply(lambda x: int(x.split("_")[1]))

        # 根据评分生成真实情绪标签
        df["true_sentiment"] = df["rating"].apply(rate_to_sentiment)

        # 判断是否预测正确
        df["correct"] = (df["true_sentiment"] == df["sentiment"]).astype(int)

        # 计算准确率
        acc = df["correct"].mean()
        print(f"{file} 的准确率: {acc * 100:.2f}%")

        results.append({"file": file, "accuracy": acc, "response_acc": df["response_err_cnt"][0]})

# 汇总结果为一个DataFrame
summary = pd.DataFrame(results)
summary.sort_values(by="accuracy", ascending=False, inplace=True)

# 保存汇总结果
summary.to_csv(os.path.join(folder3, "accuracy_summary.csv"), index=False, encoding="utf-8-sig")

print("\n所有文件处理完成，汇总结果已保存为 accuracy_summary.csv")
print(summary)

