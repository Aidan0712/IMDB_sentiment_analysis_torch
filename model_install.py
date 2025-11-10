from transformers import AutoModel, AutoTokenizer
import os

# 模型列表
model_list = [
    "microsoft/deberta-v2-xxlarge",
    "microsoft/deberta-v3-large",
    "microsoft/deberta-v3-base",
]

# 本地保存根目录
save_root = "./models"

# 确保根目录存在
os.makedirs(save_root, exist_ok=True)

for model_name in model_list:
    # 模型本地保存路径（将 / 替换为 _）
    local_dir = os.path.join(save_root, model_name.replace("/", "_"))

    # 检查模型是否已经下载（权重文件存在即可认为已下载）
    model_file = os.path.join(local_dir, "pytorch_model.bin")
    if os.path.exists(model_file):
        print(f"{model_name} 已经存在，跳过下载。")
        continue

    print(f"正在下载模型: {model_name}")
    os.makedirs(local_dir, exist_ok=True)

    # 下载模型
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(local_dir)

    # 下载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_dir)

    print(f"{model_name} 已保存到 {local_dir}\n")
print("所有模型下载完成！")
