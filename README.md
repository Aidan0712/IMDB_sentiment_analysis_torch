# IMDB_sentiment_analysis_torch

## DeBERTa-v3-base + LoRA 微调在 SST-2 上不同样本量的准确率

| 样本数 | 16 | 64 | 256 | 1024 | 4096 | 8192 | 16384 | 全量 |
|--------|------|------|--------|--------|---------|---------|----------|--------|
| **准确率** | 44.98 | 44.98 | 44.98 | 44.98 | 91.33 | 92.52 | 92.93 | 94.46 |

![alt text](<images/DeBERTa-v3-base + LoRA Accuracy on SST-2.png>)

## Ollama 本地模型的指令学习在 SST-2 上不同数据量的准确率

| 模型 | 16 | 64 | 256 | 512 |
|------|------|------|-------|------|
| **DeepSeek-R1-8B** | 87.50 | 93.75 | 90.23 | 90.63 |
| **Llama3.1-8B** | 81.25 | 90.97 | 90.36 | 89.16 |
| **Qwen3-8B** | 89.23 | 89.52 | 89.84 | 89.97 |
| **Gemma2-9B** | 90.01 | 90.00 | 90.07 | 89.79 |
| **Phi-4-mini (3.8B)** | 89.78 | 89.87 | 89.32 | 85.51 |
| **Mistral-7B** | 85.52 | 85.40 | 85.33 | 84.86 |

![alt text](<images/Ollama Local Models Accuracy on SST-2.png>)

## 11.25 进度
- 使用deberta-v3-base + lora完成在不同样本量sst-2上的微调
- 使用Ollama完成六个模型的本地部署
- 完成《动手学大模型》教程第二课《提示学习与思维链》的学习

### DeBERTa-v3-base + LoRA 微调 SST-2
#### 修改
- 以imdb_deberta_lora.py为模板，将数据集集替换为SST-2，然后处理数据适应原代码的输入格式
***（实际使用时发现dataset中的test数据label全为-1，遂使用train划分测试和训练集）***
```python
dataset = load_dataset("glue", "sst2", split="train")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
test_dataset = dataset['test']

train = pd.DataFrame({
    "id": [train_dataset[i]["idx"] for i in range(len(train_dataset))],
    "review": [train_dataset[i]["sentence"] for i in range(len(train_dataset))],
    "sentiment": [train_dataset[i]["label"] for i in range(len(train_dataset))]
})

test = pd.DataFrame({
    "id": [test_dataset[i]["idx"] for i in range(len(test_dataset))],
    "review": [test_dataset[i]["sentence"] for i in range(len(test_dataset))],
    "sentiment": [test_dataset[i]["label"] for i in range(len(test_dataset))]
})
```
#### 结果
- 使用若干小样本量(16，64，256，1024)和全量进行微调，测试数据准确率
***（测试时，小样本量和全量之间准确率跨度过大，遂加入几个中样本量，但效果一般***

| 样本数 | 16 | 64 | 256 | 1024 | 4096 | 8192 | 16384 | 全量 |
|--------|------|------|--------|--------|---------|---------|----------|--------|
| **准确率** | 44.98 | 44.98 | 44.98 | 44.98 | 91.33 | 92.52 | 92.93 | 94.46 |

#### 分析
- **小样本表现异常（16~1024）**：准确率固定在约 44.98%，接近随机猜测水平。
  - 原因猜测：模型过大，小样本无法充分训练。数据集划分后训练样本过少，难以泛化。

- **中样本量后效果显著（≥4096）**：准确率迅速提升，接近全量训练效果。
  - 原因猜测：在几千轮之后，学习率有显著上升，LoRA 微调才开始充分发挥作用。

### Ollama中本地模型的指令学习预测
#### 修改
- 使用ollama库中的generate()方法调用指定模型执行prompt
***generate对比chat的特点：无上下文记忆，每一个prompt不干扰***
```python
def call_ollama(prompt, model):
    response = ollama.generate(
        model=model,
        prompt=prompt
    )
    return response["response"]
```
- 生成prompt
```python
prompt_style = """Below is an instruction that describes a task, paired with 
an input that provides further context. Write a response that appropriately 
completes the request. 

Before answering, think carefully about the question and create a step-by-step
chain of thoughts to ensure a logical and accurate response. 

### Instruction: 
Analyze the given text from an online review and determine the sentiment 
polarity. Return a single number of either 0 and 1, with 0 being negative 
and 1 being the positive sentiment.  

### Input: 
{} 

### Response:
<think>
"""

# 对于每个语句
prompt = prompt_style.format(sentence)
```
#### 结果
- 与deberta微调相应，在每个模型的指定数据量点进行准确率计算
*（本地模型较慢，最大数据量只设置到了512）*

| 模型 | 16 | 64 | 256 | 512 |
|------|------|------|-------|------|
| **DeepSeek-R1-8B** | 87.50 | 93.75 | 90.23 | 90.63 |
| **Llama3.1-8B** | 81.25 | 90.97 | 90.36 | 89.16 |
| **Qwen3-8B** | 89.23 | 89.52 | 89.84 | 89.97 |
| **Gemma2-9B** | 90.01 | 90.00 | 90.07 | 89.79 |
| **Phi-4-mini (3.8B)** | 89.78 | 89.87 | 89.32 | 85.51 |
| **Mistral-7B** | 85.52 | 85.40 | 85.33 | 84.86 |

#### 分析 
- **小样本量时波动较大**：一些模型小幅度高于平均，一些模型全程较为平均
- **全模型总结**：
  - DeepSeek-R1-8B：表现最佳，小数据量时波动较大，最终准确率稳定在90%左右，但运行时间也是最长的
  - Llama 3.1-8B：表现较优，小数据量时略有波动，最终准确率稳定在89%左右
  - Qwen3-8B、Gemma2-9B：表现较优，小数据量波动不大，最终准确率稳定在90%左右
  - Phi-4-mini：表现还行，小数据量能稳定在90%左右，最终准确率降至86%左右，但运行速度相较7/8B的模型快很多
  - Mistral-7B：表现相对较差，准确率一直稳定在85%-86%，但运行速度不如Phi-4-mini快

### derberta微调和大模型指令学习对比分析
- **数据依赖**：DeBERTa 微调依赖中/大样本，指令学习在小样本即可达到较高性能。  
- **训练成本**：DeBERTa 需要微调，迭代多，训练时间和资源消耗大；大模型指令学习仅需生成 prompt，即可快速预测。  
- **准确率**：
  - **小样本**：DeBERTa 微调在小样本时完全失效，大模型指令学习用小样本即可高效预测。  
  - **全量样本**：DeBERTa 微调在大量数据下准确率更高，大模型指令学习表现稳定但略低。 



## 11.17 进度
- 完成Regularized Dropout和Supervised Contrastive Learning方法的运行和准确率统计
- 完成unsloth环境的配置和示例UnslothSafeTrainer封装的运行
- 在UnslothSafeTrainer的基础上，加入了compute_loss损失计算方法的重写，尝试了两种重写：
  - 选择调用适合分类任务的torch.nn中的传统CrossEntropyLoss交叉熵损失函数
  - 同时调用CrossEntropyLoss和所给losses中的SupConLoss方法，并然后加权组合输出
- 完成R-Drop和Supervised Contrastive Learning的B站论文讲解

### R-Drop
- 一句话理解：在Dropout随机丢弃神经元，提高泛化能力的基础上，调整丢弃率或引入额外约束来增强正则化

### Supervised Contrastive Learning
- 一句话理解：让相同label的距离更小，让不同label的距离更大

### unsloth
- unsloth：自动优化模型的微调过程，使得在较少参数更新的情况下，能够更快速和高效地进行模型微调。

### compute_loss调整
- 原Train调用方法```label_smoother()```：
```python
loss = self.label_smoother(outputs, labels)
```

- 重写compute_loss调用```CrossEntropyLoss()```：
```python
class UnslothSafeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = None
        self._num_labels = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels", None)

        outputs = model(**inputs)

        if labels is None:
            return super().compute_loss(model, inputs, return_outputs)

        logits = outputs.logits

        if self.loss_fn is None:
            if hasattr(model.config, 'num_labels'):
                self._num_labels = model.config.num_labels
            else:
                self._num_labels = logits.size(-1)
            self.loss_fn = nn.CrossEntropyLoss()

        if logits.size(-1) != self._num_labels:
            self._num_labels = logits.size(-1)
            self.loss_fn = nn.CrossEntropyLoss()

        loss = self.loss_fn(
            logits.contiguous().view(-1, self._num_labels),
            labels.contiguous().view(-1)
        )

        if return_outputs:
            return loss, outputs
        return loss
```

- 重写compute_loss调用```CrossEntropyLoss()```和```SupConLoss```，并授予权重：
```python
class UnslothSafeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = None
        self.sc_loss = SupConLoss(temperature=0.07)
        self._num_labels = None
        self.alpha = 0.1 # 为两个loss方法授予权重
        self.beta = 0.9


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels", None)

        outputs = model(**inputs)

        if labels is None:
            return super().compute_loss(model, inputs, return_outputs)

        logits = outputs.logits

        if self.loss_fn is None:
            if hasattr(model.config, 'num_labels'):
                self._num_labels = model.config.num_labels
            else:
                self._num_labels = logits.size(-1)
            self.loss_fn = nn.CrossEntropyLoss()

        if logits.size(-1) != self._num_labels:
            self._num_labels = logits.size(-1)
            self.loss_fn = nn.CrossEntropyLoss()

        ce_loss = self.loss_fn(
            logits.contiguous().view(-1, self._num_labels),
            labels.contiguous().view(-1)
        )

        sc_loss = self.sc_loss(logits, labels)

        total_loss = self.alpha * sc_loss + self.beta * ce_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss
```
- 结果与分析
  - 仅使用CrossEntropyLoss：相较于原Train的方法有大约0.6%的提升，说明使用传统交叉熵损失函数对二值分类任务确有提升
  - 同时使用使用CrossEntropyLoss和SupConLoss：相较于原Train的方法有微小提升，可能是SupConLoss更适用于多值分类任务，对于二值分类效果不明显



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


## 总准确率统计表格 (按准确率排序)
| **file** | **accuracy** |
| :---: | :---: |
| deberta_ptuning_base | 0.95776 |
| roberta_trainer | 0.95404 |
| deberta_lora_base | 0.95364 |
| deberta_unsloth_celoss | 0.94316 |
| bert_rdorp | 0.94028 |
| bert_scl_trainer | 0.9396 |
| deberta_unsloth_ce_sc | 0.93748 |
| bert_trainer | 0.93664 |
| deberta_unsloth | 0.93612 |
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
| lstm | 0.5 |
| capsule_lstm | 0.5 |