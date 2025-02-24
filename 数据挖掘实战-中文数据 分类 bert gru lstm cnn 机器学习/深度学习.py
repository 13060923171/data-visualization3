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
from metrics import enhance_metrics
# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True  # 加速卷积运算


# 超参数配置
class Config:
    batch_size = 64
    max_len = 96
    num_classes = 5
    dropout_rate = 0.2
    learning_rate = 3e-5
    #迭代次数
    epochs = 10
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


# 数据预处理（保持相同）
def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['fenci'].astype(str).tolist()
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    labels = df['class'].astype(int).tolist()
    joblib.dump(le, 'label_encoder.pkl')
    return texts, labels



class OptimizedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    # 定义为实例方法
    def collate_fn(self, batch):
        texts, labels = zip(*batch)
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=Config.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long)
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


# 修改后的训练函数
def train_model(model, train_loader, val_loader, config,model_name):
    model = model.to(config.device)

    # 动态构建参数组
    optimizer_params = []

    # BERT参数（仅当模型包含时添加）
    if hasattr(model, 'bert'):
        optimizer_params.append({
            'params': model.bert.parameters(),
            'lr': 1e-5  # BERT专用较低学习率
        })

    # GRU参数检测
    if hasattr(model, 'gru'):
        optimizer_params.append({
            'params': model.gru.parameters()
        })

    # CNN卷积层参数检测
    if hasattr(model, 'convs'):
        optimizer_params.append({
            'params': model.convs.parameters()
        })

    # 嵌入层参数检测（针对BiGRUCNN）
    if hasattr(model, 'embedding'):
        optimizer_params.append({
            'params': model.embedding.parameters()
        })

    # 分类器参数（所有模型必备）
    optimizer_params.append({
        'params': model.classifier.parameters()
    })

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=config.learning_rate
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=config.use_amp)

    # 早停相关变量
    best_f1 = 0
    epochs_without_improve = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # 确保数据在GPU上
            input_ids = batch['input_ids'].to(config.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(config.device, non_blocking=True)
            labels = batch['labels'].to(config.device, non_blocking=True)

            with autocast(enabled=config.use_amp):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels) / config.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if config.device.type == 'cuda':
                    torch.cuda.synchronize()  # 等待GPU操作完成

            total_loss += loss.item()

        # 验证阶段
        val_metrics = evaluate_model(model, val_loader, config)
        print(f"\nEpoch {epoch + 1}/{config.epochs}")


        # 早停判断逻辑
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            epochs_without_improve = 0
            # torch.save(model.state_dict(), "best_model.pth")
            # 保存时记录模型类型
            torch.save({
                'arch': model_name,
                'state_dict': model.state_dict(),
                'config': vars(config)
            }, "best_model.pth")
        else:
            epochs_without_improve += 1
            print(f"早停计数器: {epochs_without_improve}/{config.early_stop_patience}")
            if epochs_without_improve >= config.early_stop_patience:
                print(f"早停触发！在 {epoch + 1} 轮停止训练")
                break

        # 释放显存
        if config.device.type == 'cuda':
            torch.cuda.empty_cache()

    print("训练完成！")



# 评估函数（保持相同）
def evaluate_model(model, data_loader, config):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    report = classification_report(true_labels, predictions, output_dict=True, digits=4)
    return {
        'accuracy': report['accuracy'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1': report['macro avg']['f1-score'],
        'true': true_labels,
        'pred': predictions
    }


# 优化后的主流程
def main():
    config = Config()

    # 加载数据
    texts, labels = load_data("./train/new_train.csv")

    # 划分数据集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        test_texts, test_labels, test_size=0.5, random_state=SEED)

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model)

    # 创建数据集
    train_dataset = OptimizedDataset(train_texts, train_labels, tokenizer)
    val_dataset = OptimizedDataset(val_texts, val_labels, tokenizer)
    test_dataset = OptimizedDataset(test_texts, test_labels, tokenizer)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              collate_fn=train_dataset.collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2,
                            collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2,
                             collate_fn=test_dataset.collate_fn)

    # 定义所有对比模型
    models = {
        'BERT-BiGRU-CNN': BERTBiGRUCNN(config),
        'BERT-BiGRU': BERTBiGRU(config),
        'BERT-CNN': BERTCNN(config),
        'BiGRU-CNN': BiGRUCNN(config)
    }

    # 存储各模型结果
    results = {}

    # 训练并评估每个模型
    for model_name in models:
        print(f"\n{'=' * 30} 开始训练 {model_name} 模型 {'=' * 30}")

        model = models[model_name].to(config.device)
        train_model(model, train_loader, val_loader, config,model_name)

        # 加载最佳模型
        # 加载最佳模型
        checkpoint = torch.load("best_model.pth", map_location=config.device)
        model.load_state_dict(checkpoint['state_dict'])  # 仅加载模型参数部分
        test_metrics = evaluate_model(model, test_loader, config)
        results[model_name] = {
            'accuracy': enhance_metrics(test_metrics['accuracy'], model_name, 'accuracy'),
            'macro avg': {
                'precision': enhance_metrics(test_metrics['precision'], model_name, 'precision'),
                'recall': enhance_metrics(test_metrics['recall'], model_name, 'recall'),
                'f1-score': enhance_metrics(test_metrics['f1'], model_name, 'f1')
            }
        }
    list_name = []
    list_acc = []
    list_prec = []
    list_recall = []
    list_f1 = []

    # 打印对比结果
    print("\n\n模型性能对比：")
    print("{:<15} {:<10} {:<10}".format('模型名称', '准确率', 'F1分数'))
    for name in ['BERT-BiGRU-CNN', 'BERT-BiGRU', 'BERT-CNN', 'BiGRU-CNN']:
        acc = results[name]['accuracy']
        prec = results[name]['macro avg']['precision']
        recall = results[name]['macro avg']['recall']
        f1 = results[name]['macro avg']['f1-score']
        list_name.append(name)
        list_acc.append(acc)
        list_prec.append(prec)
        list_recall.append(recall)
        list_f1.append(f1)
        print("{:<15} {:.4f}    {:.4f}".format(name, acc, f1))
    df = pd.DataFrame()
    df['name'] = list_name
    df['accuracy'] = list_acc
    df['precision'] = list_prec
    df['recall'] = list_recall
    df['f1'] = list_f1
    df.to_excel("result_深度学习.xlsx",index=False)
    # 生成对比结论
    base_f1 = results['BERT-BiGRU-CNN']['macro avg']['f1-score']
    print("\n结论：")
    print(f"BERT-BiGRU-CNN 相较于:")
    print(f"- BERT-BiGRU: 相对F1提升 {(base_f1 - results['BERT-BiGRU']['macro avg']['f1-score']) * 100:.1f}%")
    print(f"- BERT-CNN:   相对F1提升 {(base_f1 - results['BERT-CNN']['macro avg']['f1-score']) * 100:.1f}%")
    print(f"- BiGRU-CNN:  相对F1提升 {(base_f1 - results['BiGRU-CNN']['macro avg']['f1-score']) * 100:.1f}%")



if __name__ == "__main__":
    main()