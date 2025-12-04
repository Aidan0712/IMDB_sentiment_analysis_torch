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

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_flag = True

model_local_flag = True

max_seq_length = 1024

os.environ["UNSLOTH_OFFLINE"] = "1"

train = pd.read_csv("./word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./word2vec-nlp-tutorial/testData.tsv", header=0, delimiter="\t", quoting=3)

"""
Before answering, think carefully about the question and create a step-by-step
chain of thoughts to ensure a logical and accurate response. 
"""

# --- Alpaca 指令模版 ---
alpaca_prompt = """
### Instruction:
Below is an instruction that describes a task, 
paired with an input that provides further context. 
Write a response that appropriately completes the request.
Analyze the sentiment of the following movie review. 
Return a single word of either ‘positive’ or ‘negative’.

### Input:
{}

### Response:
{}
"""

MODELS_ALL = [
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",

    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",

    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

    "unsloth/Qwen3-4B-unsloth-bnb-4bit",

    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit",
]

MODELS = [
    "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit",
]

model_result_map = {
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit": "llama3.2_1b",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit": "gemma3_1b",
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit": "qwen3_1.7b",

    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit": "mistral_7b",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit": "llama3.2_3b",
    "unsloth/Qwen3-4B-unsloth-bnb-4bit": "qwen3_4b",

    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit": "gemma3_4b",
    "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit": "phi4_mini_3.8B",
}

# --- 数据格式化函数 ---
def formatting_prompts_func(examples):
    inputs = examples["text"]
    labels = examples["label"]
    outputs_text = []

    global EOS_TOKEN, max_seq_length
    if EOS_TOKEN is None:
        raise ValueError("EOS_TOKEN is not set. Make sure tokenizer is loaded first.")

    for input_text, label in zip(inputs, labels):
        reserve_tokens = 30
        available_len = max_seq_length - len(alpaca_prompt) - reserve_tokens
        input_text = input_text[:available_len] if len(input_text) > available_len else input_text
        label_text = "positive" if label == 1 else "negative"
        text = alpaca_prompt.format(input_text, label_text) + EOS_TOKEN
        outputs_text.append(text)

    return {"text": outputs_text}


if __name__ == '__main__':
    org_time = time.time()

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    logger.info(r"running %s" % ''.join(sys.argv))

    last_time = time.time()

    logger.info("Loading data...")

    # test begin
    if test_flag:
        train = train[0:500]
        test = test[0:50]
    else:
        test = test[0:5000]
    # test end

    train, val = train_test_split(train, test_size=.2, random_state=3407)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    logger.info(f"Used time: {time.time() - last_time}")

    for model_name in MODELS:
        last_time = time.time()

        logger.info("Loading model...")

        # model_local
        if model_local_flag:
            model_path = os.path.join("/root/autodl-tmp/huggingface/", model_name)
        else:
            model_path = model_name
        # model_local

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=True,
            max_seq_length=max_seq_length,
            dtype=None,
            local_files_only=True
        )

        logger.info(f"Used time: {time.time() - last_time}")

        EOS_TOKEN = tokenizer.eos_token
        logger.info(f"EOS token set to: {EOS_TOKEN}")

        logger.info(f"tokenizer.model_max_length：{tokenizer.model_max_length}")

        logger.info("Setting up PEFT...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_gradient_checkpointing="unsloth",
            max_seq_length=max_seq_length,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
        )

        logger.info("Model parameters:" + str(sum(p.numel() for p in model.parameters())))

        logger.info("Formatting datasets...")
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True, num_proc=4)
        val_dataset = val_dataset.map(formatting_prompts_func, batched=True, num_proc=4)

        logger.info("Setting up Training Arguments...")
        training_args = TrainingArguments(
            output_dir="model_output",
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

        sft_config = SFTConfig(
            output_dir="model_output",
            per_device_train_batch_size=16,
            # gradient_accumulation_steps=2,
            warmup_steps=100,
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

        logger.info("Initializing SFTTrainer...")

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_config,
        )

        last_time = time.time()
        logger.info("Starting training...")
        trainer_stats = trainer.train()
        print(trainer_stats)
        logger.info(f"Used time: {time.time() - last_time}")

        logger.info("Saving model...")
        model_ab = model_result_map.get(model_name, 'default')
        model_save_path = f"./models_save/{model_ab}"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)


        last_time = time.time()
        logger.info("Starting inference...")
        FastLanguageModel.for_inference(model)

        inference_prompt = """Below is an instruction that describes a task, 
        paired with an input that provides further context. 
        Write a response that appropriately completes the request.

        ### Instruction:
        Analyze the sentiment of the following movie review. 
        Return a single word of either ‘positive’ or ‘negative’.

        ### Input:
        {}

        ### Response:"""


        test_texts = test_dataset['text']
        test_ids = test['id']
        predictions = []
        response_err_cnt = 0
        batch_size = 1

        for i in range(0, len(test_texts), batch_size):
            if i % 256 == 0:
                print()
                print(f"Batch {i} Inference")
            batch = test_texts[i:i + batch_size]
            prompt = []
            for review_text in batch:
                if review_text is None:
                    review_text = ""
                review_text = str(review_text)
                reserve_tokens = 30
                available_len = max_seq_length - len(inference_prompt) - reserve_tokens
                review_text = review_text[:available_len] if len(review_text) > available_len else review_text
                prompt.append(inference_prompt.format(review_text))
            # inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for out in generated_text:
                try:
                    response_part = out.split("### Response:")[1].strip().lower()
                    if "positive" in response_part:
                        predictions.append(1)
                    elif "negative" in response_part:
                        predictions.append(0)
                    else:
                        print(i, end=' ')
                        response_err_cnt += 1
                        predictions.append(-1)
                except (IndexError, AttributeError):
                    logger.warning(f"Failed to parse model output: {generated_text}")
                    print()
                    response_err_cnt += 1
                    predictions.append(-1)

        print()
        logger.info(f"response_err_cnt：{response_err_cnt}")
        logger.info(f"Used time: {time.time() - last_time}")
        logger.info("Saving results...")

        result_output = pd.DataFrame(data={"id": test_ids, "sentiment": predictions,
                                           "response_err_cnt": response_err_cnt, })
        os.makedirs("./results_instruction_tuning", exist_ok=True)
        result_output.to_csv(f"./results_instruction_tuning/unsloth_lora_{model_ab}.csv", index=False, quoting=3)
        logger.info('Result saved!')

        del model
        del tokenizer
        torch.cuda.empty_cache()

