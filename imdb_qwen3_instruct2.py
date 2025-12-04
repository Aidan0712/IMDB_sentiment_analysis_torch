import os
import sys
import logging
import pandas as pd

import unsloth
import numpy as np
import torch
import evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split

from unsloth import FastLanguageModel

from transformers import TrainingArguments, AutoModelForSequenceClassification
from trl import SFTTrainer, SFTConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_flag = True

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train = pd.read_csv("./word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./word2vec-nlp-tutorial/testData.tsv", header=0, delimiter="\t", quoting=3)

# --- Alpaca 指令模版 ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the sentiment of the following movie review. Respond with 'positive' or 'negative'.

### Input:
{}

### Response:
{}"""


# --- 数据格式化函数 ---
def formatting_prompts_func(examples):
    inputs = examples["text"]
    labels = examples["label"]
    outputs_text = []

    global EOS_TOKEN
    if EOS_TOKEN is None:
        raise ValueError("EOS_TOKEN is not set. Make sure tokenizer is loaded first.")

    for input_text, label in zip(inputs, labels):
        label_text = "positive" if label == 1 else "negative"
        text = alpaca_prompt.format(input_text, label_text) + EOS_TOKEN
        outputs_text.append(text)

    return {"text": outputs_text}


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    logger.info(r"running %s" % ''.join(sys.argv))

    logger.info("Loading data...")

    # test begin
    if test_flag:
        train = train[0:500]
        test = test[0:50]
    # test end

    train, val = train_test_split(train, test_size=.2, random_state=3407)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    # --- 2. 加载模型和 Tokenizer ---
    logger.info("Loading model...")
    model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        max_seq_length=512,
        dtype=None,
    )

    EOS_TOKEN = tokenizer.eos_token
    logger.info(f"EOS token set to: {EOS_TOKEN}")

    logger.info("Setting up PEFT...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_gradient_checkpointing="unsloth",
        max_seq_length=512,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
    )

    logger.info("Model parameters:" + str(sum(p.numel() for p in model.parameters())))

    # --- 4. 格式化数据集 ---
    logger.info("Formatting datasets...")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True, num_proc=4)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True, num_proc=4)

    # --- 5. 训练参数 ---
    logger.info("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir="outputs_qwen",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="epoch",
        eval_strategy="epoch",
    )

    # --- 6. 初始化 SFTTrainer ---
    logger.info("Initializing SFTTrainer...")

    sft_config = SFTConfig(
        output_dir="outputs_qwen",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="epoch",
        eval_strategy="epoch",

        dataset_num_proc=4,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )

    # --- 7. 训练 ---
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    print(trainer_stats)

    # --- 8. 推理 (Generation) ---
    logger.info("Starting inference...")
    FastLanguageModel.for_inference(model)

    inference_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    Analyze the sentiment of the following movie review. Respond with 'positive' or 'negative'.
    ### Input:
    {}
    ### Response:
    """

    test_texts = test_dataset['text']
    test_ids = test['id']
    predictions = []

    for review_text in test_texts:
        prompt = inference_prompt.format(review_text)
        inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=1024).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            response_part = generated_text.split("### Response:")[1].strip().lower()

            if "positive" in response_part:
                predictions.append(1)
            elif "negative" in response_part:
                predictions.append(0)
            else:
                predictions.append(0)

        except (IndexError, AttributeError):
            logger.warning(f"Failed to parse model output: {generated_text}")
            predictions.append(0)

    # --- 9. 保存结果 ---
    logger.info("Saving results...")
    result_output = pd.DataFrame(data={"id": test_ids, "sentiment": predictions})
    os.makedirs("./results_instruction_tuning", exist_ok=True)
    result_output.to_csv("./results_instruction_tuning/qwen3_4b_instruct_unsloth.csv", index=False, quoting=3)
    logger.info('Qwen result saved!')