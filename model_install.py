import os
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODELS_ALL = [
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",

    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]

MODELS = [
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",

    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]


def preload_model(model_name):
    """预加载模型到缓存"""
    try:
        logger.info(f"正在缓存: {model_name}")
        local_path = snapshot_download(repo_id=model_name, local_dir=None, resume_download=True)
        logger.info(f"模型已缓存到: {local_path}")
        return True

    except Exception as e:
        return False


if __name__ == "__main__":
    logger.info("开始预加载Unsloth模型到缓存...")

    for model in MODELS:
        preload_model(model)
        print("-" * 50)

    logger.info(f"预加载完成！")