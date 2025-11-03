import subprocess
import os

base_dir = r"D:\2025Autumn\Scientific Research Training\Task3\imdb_sentiment_analysis_torch"

scripts = [
    # "imdb_cnn.py",
    # "imdb_lstm.py",
    # "imdb_gru.py",
    # "imdb_cnnlstm.py",
    # "imdb_attention_lstm.py",
    # "imdb_capsule_lstm.py",


    "imdb_transformer.py",
    # "imdb_bert_trainer.py",
    # "imdb_bert_native.py",
    # "imdb_bert_scratch.py",
    # "imdb_roberta_trainer.py",

    # "imdb_distilbert_trainer.py",
    # "imdb_distilbert_native.py",
]

# 运行循环
for script in scripts:
    script_path = os.path.join(base_dir, script)
    print(f"\n🚀 正在运行：{script_path}\n{'=' * 60}")

    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    # 保存日志
    log_name = os.path.splitext(script)[0] + ".log"
    log_path = os.path.join(base_dir, "logs", log_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    stdout_text = result.stdout or ""
    stderr_text = result.stderr or ""

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(stdout_text)
        f.write("\n\n[ERRORS]\n")
        f.write(stderr_text)

    print(f"✅ {script} 已完成，日志已保存到 {log_path}\n")

print("\n🎯 所有脚本运行完成！")
