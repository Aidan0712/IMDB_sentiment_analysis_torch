import matplotlib.pyplot as plt

# ------------------------------
# DeBERTa-v3-base + LoRA
# ------------------------------
samples_deberta = [16, 64, 256, 1024, 4096, 8192, 16384, 'all']
accuracy_deberta = [44.98, 44.98, 44.98, 44.98, 91.33, 92.52, 92.93, 94.46]

plt.figure(figsize=(8,5))
plt.plot(samples_deberta, accuracy_deberta, marker='o', linestyle='-', color='blue')
plt.title("DeBERTa-v3-base + LoRA Accuracy on SST-2")
plt.xlabel("Number of Samples")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()

# Y轴自适应
plt.ylim(min(accuracy_deberta)-1, max(accuracy_deberta)+1)

plt.savefig("deberta_lora_sst2_adaptive.png", dpi=300)
plt.show()


# ------------------------------
# Ollama Local Models
# ------------------------------
samples_ollama = [16, 64, 256, 512]
accuracy_ollama = {
    "DeepSeek-R1-8B": [87.50, 93.75, 90.23, 90.63],
    "Llama3.1-8B": [81.25, 90.97, 90.36, 89.16],
    "Qwen3-8B": [89.23, 89.52, 89.84, 89.97],
    "Gemma2-9B": [90.01, 90.00, 90.07, 89.79],
    "Phi-4-mini (3.8B)": [89.78, 89.87, 89.32, 85.51],
    "Mistral-7B": [85.52, 85.40, 85.33, 84.86]
}

plt.figure(figsize=(10,6))
for model, acc in accuracy_ollama.items():
    plt.plot(samples_ollama, acc, marker='o', linestyle='-', label=model)
plt.title("Ollama Local Models Accuracy on SST-2")
plt.xlabel("Number of Samples")
plt.ylabel("Accuracy (%)")
plt.xticks(samples_ollama)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Y轴自适应
all_acc = [v for acc_list in accuracy_ollama.values() for v in acc_list]
plt.ylim(min(all_acc)-1, max(all_acc)+1)

plt.savefig("ollama_sst2_adaptive_labels.png", dpi=300)
plt.show()
