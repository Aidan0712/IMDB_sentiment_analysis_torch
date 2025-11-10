# IMDB_sentiment_analysis_torch

## 本周准确率表格（按准确率排序）
| **file** | **accuracy** |
| :---: | :---: |
| deberta_ptuning_base | 0.95776 |
| deberta_lora_base | 0.95364 |
| deberta_prompt_base | 0.51532 |

## 总准确率统计表格 (按准确率排序)
| **file** | **accuracy** |
| :---: | :---: |
| deberta_ptuning_base | 0.95776 |
| roberta_trainer | 0.95404 |
| deberta_lora_base | 0.95364 |
| bert_trainer | 0.93664 |
| bert_scratch | 0.93352 |
| distilbert_trainer | 0.92828 |
| distilbert_native | 0.91252 |
| bert_native | 0.90088 |
| attention_lstm | 0.86168 |
| cnnlstm | 0.83648 |
| transformer | 0.83208 |
| cnn | 0.77256 |
| gru | 0.76368 |
| deberta_prompt_base | 0.51532 |
| capsule_lstm | 0.5 |
| lstm | 0.5 |


## 11.10 进度
- 使用deberta_v3_base模型完成lora、p-tuning和prompt方法的运行和准确率统计，prefix方法似乎无法在deberta中使用
- 完成lora原理部分的学习
### 通用调整
- 数据导入时加入了最大长度限制```max_length=512```
```python
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)
```
- TrainingArguments使用```gradient_accumulation_steps=8```提升batch_size，开启半精度训练```bf16 = True```
```python
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

    gradient_accumulation_steps=8,
    bf16 = True,
    dataloader_num_workers = 2,
    dataloader_pin_memory=True,
)
```
- 启用梯度检查点```gradient_checkpointing_enable```，启用输入梯度需求```enable_input_require_grads```
```python
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
```


### 特殊调整
#### lora
- 开启指定的适配器模块```target_modules=["query_proj", "value_proj"]```
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_proj", "value_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
```
#### p_tuning
- TrainingArguments中指定较高的学习率```learning_rate=5e-4```
```python
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

    learning_rate=5e-4,
    gradient_accumulation_steps=8,
    fp16=True,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)
```
- 增加tokens数量```num_virtual_tokens=30```
```python
peft_config = PromptEncoderConfig(
    num_virtual_tokens=30,
    encoder_hidden_size=128,
    task_type=TaskType.SEQ_CLS
)
```
#### prompt
- TrainingArguments中指定较高的学习率```learning_rate=1e-3```
```python
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
    fp16 = True,
    dataloader_num_workers = 2,
    dataloader_pin_memory=True,
)
```
- 增加tokens数量```num_virtual_tokens=50```
```python
peft_config = PromptTuningConfig(
    num_virtual_tokens=50,
    task_type=TaskType.SEQ_CLS
)
```

### 总结
- lora与p-tuning准确率符合预期，且在差不多的准确率下，lora训练时间更短
- prompt准确率不理想，可能是有错误未发现，也可能prompt并不适合deberta
- DeBERTa是encoder-only，不支持past_key_values，所以prefix似乎无法在deberta使用

## 11.3 进度
- 完成所有方法的运行与准确率统计，统计数据源文件为./result/accuracy_summary.csv
- 完成bert原理视频的学习

## 10.26 进度
- 完成cnn、gru、lstm、attention_lstm和capsule_lstm的运行和准确率统计，统计数据源文件为./result/accuracy_summary.csv
- transformer报错修改还未完成
- bert运行时显卡过热关机，考虑使用云端服务器或去修理
- 其余还未进行运行
