import os
import sys
import logging
import math
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 读取数据
train = pd.read_csv("./word2vec-nlp-tutorial/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("./word2vec-nlp-tutorial/testData.tsv", header=0,
                   delimiter="\t", quoting=3)


# 构建词汇表
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.idx = 2

    def build_vocab(self, texts, min_freq=1):
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        for word, freq in word_freq.items():
            if freq >= min_freq:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, max_sequence_length=512):
        tokens = text.split()
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        if len(sequence) > max_sequence_length:
            sequence = sequence[:max_sequence_length]
        return sequence


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length=5000):
        super().__init__()
        self.d_model = d_model

        # 创建位置编码矩阵
        pe = torch.zeros(max_sequence_length, d_model)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_sequence_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # 只取前seq_len个位置编码
        x = x + self.pe[:, :seq_len, :]
        return x


# Transformer分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                 num_classes=2, max_sequence_length=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 位置编码
        self.position_embedding = PositionalEncoding(d_model, max_sequence_length)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True  # 使用batch_first格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()

        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        x = self.position_embedding(x)  # add pos encoding
        x = self.dropout(x)

        # 创建注意力掩码（可选）
        if lengths is not None:
            # 创建padding mask
            mask = (torch.arange(seq_len).expand(batch_size, seq_len).to(x.device) >= lengths.unsqueeze(1))
            # Transformer需要的是key_padding_mask，格式为(batch_size, seq_len)
            mask = mask
        else:
            mask = None

        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # (batch_size, seq_len, d_model)

        # 使用第一个token的输出或平均池化
        if lengths is not None:
            # 使用每个序列最后一个有效token的输出
            idx = (lengths - 1).view(-1, 1).expand(-1, self.d_model).unsqueeze(1)
            pooled_output = x.gather(1, idx).squeeze(1)
        else:
            # 使用第一个token的输出
            pooled_output = x[:, 0, :]

        # 分类
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)

        return logits


# 数据集类
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_sequence_length=2000):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 将文本转换为序列
        sequence = self.vocab.text_to_sequence(text, self.max_sequence_length)
        sequence = torch.tensor(sequence, dtype=torch.long)

        return {'text': sequence, 'label': label}


def collate_fn(batch):
    texts = [sample['text'] for sample in batch]
    labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.long)

    # 对序列进行padding
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)

    # 计算每个序列的实际长度（用于masking）
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)

    return texts_padded, lengths, labels


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 准备数据
    train_texts, train_labels, test_texts = [], [], []

    for i, review in enumerate(train["review"]):
        train_texts.append(review)
        train_labels.append(train['sentiment'][i])

    for review in test['review']:
        test_texts.append(review)

    # 构建词汇表
    vocab = Vocabulary()
    vocab.build_vocab(train_texts, min_freq=2)
    logging.info("vocab size: %d" % len(vocab.word2idx))

    # 分割训练集和验证集
    from sklearn.model_selection import train_test_split

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42
    )

    logging.info("train %d, val %d, test %d" % (len(train_texts), len(val_texts), len(test_texts)))

    # 设置最大序列长度
    max_sequence_length = 2000
    logging.info("max sequence length (capped): %d" % max_sequence_length)

    # 创建数据集
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_sequence_length)
    val_dataset = IMDBDataset(val_texts, val_labels, vocab, max_sequence_length)
    test_dataset = IMDBDataset(test_texts, [0] * len(test_texts), vocab, max_sequence_length)  # 测试标签设为0

    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 初始化模型
    vocab_size = len(vocab.word2idx)
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=256,  # 减小模型尺寸以节省内存
        nhead=8,
        num_layers=4,  # 减少层数
        num_classes=2,
        max_sequence_length=5000,  # 增加最大序列长度
        dropout=0.1
    )
    model.to(device)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        total_train_loss = 0.0
        total_train_acc = 0.0
        train_batches = 0

        # 训练阶段
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
            for batch in train_loader:
                batch_inputs, batch_lengths, batch_labels = batch
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                batch_lengths = batch_lengths.to(device)

                optimizer.zero_grad()

                # 前向传播
                outputs = model(batch_inputs, batch_lengths)  # (batch, num_class) as log_probs
                loss = criterion(outputs, batch_labels)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()

                # 计算准确率
                preds = torch.argmax(outputs, dim=1)
                acc = accuracy_score(batch_labels.cpu().numpy(), preds.cpu().numpy())

                total_train_loss += loss.item()
                total_train_acc += acc
                train_batches += 1

                # 更新进度条
                pbar.set_postfix({
                    'train loss': f'{total_train_loss / train_batches:.4f}',
                    'train acc': f'{total_train_acc / train_batches:.4f}'
                })
                pbar.update(1)

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_inputs, batch_lengths, batch_labels = batch
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                batch_lengths = batch_lengths.to(device)

                outputs = model(batch_inputs, batch_lengths)
                loss = criterion(outputs, batch_labels)

                preds = torch.argmax(outputs, dim=1)
                acc = accuracy_score(batch_labels.cpu().numpy(), preds.cpu().numpy())

                total_val_loss += loss.item()
                total_val_acc += acc
                val_batches += 1

        epoch_time = time.time() - start_time

        # 打印epoch结果
        avg_train_loss = total_train_loss / train_batches
        avg_train_acc = total_train_acc / train_batches
        avg_val_loss = total_val_loss / val_batches
        avg_val_acc = total_val_acc / val_batches

        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')
        print(f'  Time: {epoch_time:.2f}s')
        print()

    # 测试阶段
    model.eval()
    test_preds = []

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing') as pbar:
            for batch in test_loader:
                batch_inputs, batch_lengths, _ = batch
                batch_inputs = batch_inputs.to(device)
                batch_lengths = batch_lengths.to(device)

                outputs = model(batch_inputs, batch_lengths)
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy().tolist())

                pbar.update(1)

    # 保存结果
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_preds})
    os.makedirs("./result", exist_ok=True)
    result_output.to_csv("./result/transformer.csv", index=False, quoting=3)
    logging.info('Result saved!')