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

from peft import PeftModel, PeftConfig


import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_flag = True

model_local_flag = False

lora_local_flag = False

max_seq_length = 1024



test = pd.read_csv("./word2vec-nlp-tutorial/testData.tsv", header=0, delimiter="\t", quoting=3)


MODELS_ALL = [
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",

    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",

    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",

    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
]

MODELS = [
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",


    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
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
        test = test[0:500]
    else :
        test = test[0:5000]
    # test end

    test_dict = {"text": test['review']}

    test_dataset = Dataset.from_dict(test_dict)

    logger.info(f"Used time: {time.time() - last_time}")

    for model_name in MODELS:
        last_time = time.time()

        logger.info("Loading model...")

        # model_local
        if model_local_flag:
            os.environ["UNSLOTH_OFFLINE"] = "1"
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

        EOS_TOKEN = tokenizer.eos_token
        logger.info(f"EOS token set to: {EOS_TOKEN}")

        logger.info(f"Used time: {time.time() - last_time}")

        logger.info("Setting up PEFT...")

        # lora_local
        model_ab = model_result_map.get(model_name, 'default')
        if lora_local_flag:
            lora_path = f"./models_save/{model_ab}"
            model = PeftModel.from_pretrained(model, lora_path)
        else:
            print()
            # model = FastLanguageModel.get_peft_model(
            #     model,
            #     r=16,
            #     lora_alpha=32,
            #     lora_dropout=0,
            #     bias="none",
            #     random_state=3407,
            #     use_gradient_checkpointing="unsloth",
            #     max_seq_length=max_seq_length,
            #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
            #                     "gate_proj", "up_proj", "down_proj", ],
            # )

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
                                           "response_err_cnt": response_err_cnt,})
        os.makedirs("./results_instruction_tuning", exist_ok=True)
        result_output.to_csv(f"./results_instruction_tuning/unsloth_lora_{model_ab}.csv", index=False, quoting=3)
        logger.info('Result saved!')

        del model
        del tokenizer
        torch.cuda.empty_cache()

