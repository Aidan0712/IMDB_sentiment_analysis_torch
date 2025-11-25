import subprocess
import os

base_dir = os.path.dirname(__file__)

scripts = [
    # "imdb_cnn.py",
    # "imdb_lstm.py",
    # "imdb_gru.py",
    # "imdb_cnnlstm.py",
    # "imdb_attention_lstm.py",
    # "imdb_capsule_lstm.py",


    # "imdb_transformer.py",
    # "imdb_bert_trainer.py",
    # "imdb_bert_native.py",
    # "imdb_bert_scratch.py",
    # "imdb_roberta_trainer.py",

    # "imdb_distilbert_trainer.py",
    # "imdb_distilbert_native.py",

    # "imdb_deberta_lora.py",
    # "imdb_deberta_prompt.py",
    # "imdb_deberta_prefix.py",
    # "imdb_deberta_ptuning.py",

    # "imdb_modernbert_unsloth.py",
    # "imdb_bert_rdrop.py",
    # "imdb_bert_scl_trainer.py",
    # "imdb_modernbert_unsloth_celoss.py"

    "imdb_sst2_deberta_loop.py",
    # "imdb_sst2_deberta.py",
    # "imdb_ollama_loop.py",
]
# 运行循环
for script in scripts:
    script_path = os.path.join(base_dir, script)
    print(f"\n🚀 正在运行：{script_path}\n{'=' * 60}")

    log_name = os.path.splitext(script)[0] + ".log"
    log_path = os.path.join(base_dir, "logs", log_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 打开日志文件
    with open(log_path, "w", encoding="utf-8") as log_file:
        # 使用 Popen 可以实时读取输出
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将 stderr 合并到 stdout
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="ignore"
        )

        # 实时输出到控制台并写入日志
        for line in process.stdout:
            print(line, end="")  # 输出到控制台
            log_file.write(line)  # 写入日志

        process.wait()

    print(f"\n✅ {script} 已完成，日志已保存到 {log_path}\n")

print("\n🎯 所有脚本运行完成！")
