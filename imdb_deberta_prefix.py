import os
import sys
import logging
import datasets
import evaluate

import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from peft import PrefixTuningConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.model_selection import train_test_split

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_flag = True

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

train = pd.read_csv("./word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./word2vec-nlp-tutorial/testData.tsv", header=0, delimiter="\t", quoting=3)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # test begin
    if test_flag:
        train = train.sample(100, random_state=42).reset_index(drop=True)
        test = test.sample(10, random_state=42).reset_index(drop=True)
    # test end

    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    batch_size = 32

    # model_id = "microsoft/deberta-v2-xxlarge"

    model_id = "microsoft/deberta-v3-large"

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)


    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
    )

    # Define LoRA Config
    peft_config = PrefixTuningConfig(
        num_virtual_tokens=20,
        task_type=TaskType.SEQ_CLS
    )

    # prepare model for kbit training
    # model = prepare_model_for_kbit_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    metric = evaluate.load("accuracy")

    # add
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=4,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=1000,
        save_strategy="no",
        eval_strategy="epoch",

        learning_rate=1e-3,
        gradient_accumulation_steps=8,
        bf16 = True,
        dataloader_num_workers = 2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # tokenizer=tokenizer,
        processing_class=tokenizer,
    )

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs.predictions, axis=-1).flatten()
    print("test_pred:", test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/deberta_prefix_base.csv", index=False, quoting=3)
    logging.info('result saved!')