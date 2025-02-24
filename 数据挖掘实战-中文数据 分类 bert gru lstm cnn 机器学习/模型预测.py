# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import joblib
# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True  # 加速卷积运算

# 超参数配置
# 超参数配置（修正版）
class Config:
    batch_size = 64
    max_len = 96
    num_classes = 5
    dropout_rate = 0.2
    learning_rate = 3e-5
    epochs = 3
    bert_model = "hfl/chinese-bert-wwm-ext"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_accum_steps = 2
    early_stop_patience = 5
    use_amp = True

    # 结构化参数组
    cnn = {
        'filter_num': 64,
        'filter_sizes': [3, 4, 5]
    }

    gru = {
        'hidden_size': 128,
        'num_layers': 1
    }


class BERTBiGRUCNN(nn.Module):
    """BERT+BiGRU+CNN融合模型"""

    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.bert_model)
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=config.gru['hidden_size'],
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=2 * config.gru['hidden_size'],
                out_channels=config.cnn['filter_num'],
                kernel_size=fs
            ) for fs in config.cnn['filter_sizes']  # 修正这里
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(len(config.cnn['filter_sizes']) * config.cnn['filter_num'], config.num_classes)
        )

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask).last_hidden_state
        gru_out, _ = self.gru(bert_out)
        gru_out = gru_out.permute(0, 2, 1)  # [batch, channels, seq_len]

        conved = [F.relu(conv(gru_out)) for conv in self.convs]
        # 修改池化操作
        pooled = [F.max_pool1d(conv, kernel_size=conv.size(2)).squeeze(2) for conv in conved]

        return self.classifier(torch.cat(pooled, dim=1))

class BERTBiGRU(nn.Module):
    """仅使用BERT+BiGRU"""

    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.bert_model)
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=config.gru['hidden_size'],
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(2 * config.gru['hidden_size'], config.num_classes)
        )

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask).last_hidden_state
        gru_out, _ = self.gru(bert_out)
        return self.classifier(gru_out[:, -1, :])


class BERTCNN(nn.Module):
    """仅使用BERT+CNN"""

    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.bert_model)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.bert.config.hidden_size,
                out_channels=config.cnn['filter_num'],
                kernel_size=fs
            ) for fs in config.cnn['filter_sizes']
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(len(config.cnn['filter_sizes']) * config.cnn['filter_num'], config.num_classes)
        )

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask).last_hidden_state
        bert_out = bert_out.permute(0, 2, 1)  # [batch, channels, seq_len]

        conved = [F.relu(conv(bert_out)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, kernel_size=conv.size(2)).squeeze(2) for conv in conved]

        return self.classifier(torch.cat(pooled, dim=1))


class BiGRUCNN(nn.Module):
    """仅使用BiGRU+CNN（不使用BERT）"""

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(21128, 256)  # 与BERT的hidden_size一致
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=config.gru['hidden_size'],
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=2 * config.gru['hidden_size'],
                out_channels=config.cnn['filter_num'],
                kernel_size=fs
            ) for fs in config.cnn['filter_sizes']
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(len(config.cnn['filter_sizes']) * config.cnn['filter_num'], config.num_classes)
        )

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        gru_out, _ = self.gru(emb)
        gru_out = gru_out.permute(0, 2, 1)

        conved = [F.relu(conv(gru_out)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, kernel_size=conv.size(2)).squeeze(2) for conv in conved]

        return self.classifier(torch.cat(pooled, dim=1))

def predict(texts, model_path='best_model.pth'):
    """中文文本分类预测函数
    Args:
        texts: list[str], 需要预测的文本列表
        model_path: str, 模型保存路径
    Returns:
        predictions: list[str], 预测的类别标签
        probabilities: list[dict], 每个类别的概率分布
    """
    # 初始化配置
    config = Config()
    device = config.device

    # 加载LabelEncoder
    le = joblib.load('label_encoder.pkl')

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model)

    try:
        # 加载保存的模型检查点
        checkpoint = torch.load(model_path, map_location=device)

        # 动态创建对应模型结构
        model_classes = {
            'BERT-BiGRU-CNN': BERTBiGRUCNN,
            'BERT-BiGRU': BERTBiGRU,
            'BERT-CNN': BERTCNN,
            'BiGRU-CNN': BiGRUCNN
        }
        model_class = model_classes[checkpoint['arch']]
        model = model_class(Config())
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")

    # 文本预处理
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.max_len,
        return_tensors="pt"
    )

    # 创建数据集
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # 转换为标签
    predictions = le.inverse_transform(preds)

    # 构建概率字典
    class_names = le.classes_
    probabilities = [
        {class_names[i]: float(prob[i]) for i in range(len(class_names))}
        for prob in probs
    ]

    return predictions, probabilities


# 使用示例
if __name__ == "__main__":
    # 示例文本（替换为实际需要预测的文本）
    df = pd.read_excel('test/预测文本.xlsx')
    # 设置每个块的大小
    chunk_size = 1000
    list_pd = []
    # 分块处理数据
    for start_idx in range(0, len(df), chunk_size):
        # 计算当前块的结束索引
        end_idx = start_idx + chunk_size
        # 提取当前块的数据
        chunk = df.iloc[start_idx:end_idx]
        test_texts = chunk['fenci'].tolist()
        list_label = []
        try:
            labels, probs = predict(test_texts)
            for text, label, prob in zip(test_texts, labels, probs):
                list_label.append(label)

        except Exception as e:
            print(f"预测出错: {str(e)}")

        chunk['label'] = list_label
        list_pd.append(chunk)

    data = pd.concat(list_pd,axis=0)
    data.to_excel('预测文本.xlsx', index=False)