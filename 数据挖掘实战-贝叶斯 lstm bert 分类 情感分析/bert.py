import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    precision_recall_curve
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import os

# 配置参数
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
BERT_MODEL_NAME = 'bert-base-chinese'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化环境
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
os.makedirs("./BERT_PyTorch", exist_ok=True)
# 新增标签编码
label_encoder = LabelEncoder()

# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }


# 数据加载函数
def load_data():
    # 加载原始数据
    data1 = pd.read_csv('./train/新_博文表.csv')
    data2 = pd.read_csv('./train/新_评论表.csv')

    # 合并数据
    df = pd.concat([
        data1[['fenci', 'label']],
        data2[['fenci', 'label']]
    ], axis=0)

    # 数据清洗
    df = df.dropna()
    df = df[df['fenci'].str.len() > 0]
    df['label'] = label_encoder.fit_transform(df['label'])

    return df, data1, data2



# 模型训练函数
def train_model():
    # 加载数据
    df, data1, data2 = load_data()

    # 划分数据集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['fenci'], df['label'], test_size=0.3, random_state=42
    )

    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 创建数据加载器
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model = model.to(DEVICE)

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 训练循环
    best_accuracy = 0
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        model.train()
        current_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            model.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            current_loss += loss.item()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = current_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证评估
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        accuracy = accuracy_score(true_labels, predictions)
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n')  # 新增验证损失输出

        # 保存最佳模型
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), './BERT_PyTorch/best_model.bin')
            best_accuracy = accuracy

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, 'b-o', label='训练损失', color='#1f77b4', linewidth=2)
    plt.plot(range(1, EPOCHS + 1),val_losses,'s--',label='验证损失',color='#ff7f0e',linewidth=2)
    plt.title("训练损失曲线")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # 标记最佳epoch
    best_epoch = np.argmin(val_losses)
    plt.axvline(best_epoch + 1, color='red', linestyle=':', linewidth=1.5,
                label=f'最佳epoch ({best_epoch + 1})')
    plt.legend()
    # 保存输出
    plt.tight_layout()
    plt.savefig('./BERT_PyTorch/training_loss.png')
    plt.close()

    # 最终评估
    evaluate_model(model, test_loader)
    save_predictions(model, tokenizer, data1, data2)
    plot_emotion_distribution()


# 模型评估函数
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    # 计算指标
    metrics = {
        'Accuracy': accuracy_score(true_labels, predictions),
        'Precision': precision_score(true_labels, predictions),
        'Recall': recall_score(true_labels, predictions),
        'F1': f1_score(true_labels, predictions)
    }

    # 绘制指标图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'])
    plt.ylim(0, 1)
    plt.title("模型性能指标")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.savefig('./BERT_PyTorch/performance_metrics.png')
    plt.close()

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig('./BERT_PyTorch/roc_curve.png')
    plt.close()

    # 绘制PR曲线
    precision, recall, _ = precision_recall_curve(true_labels, probabilities)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR曲线')
    plt.legend(loc="upper right")
    plt.savefig('./BERT_PyTorch/pr_curve.png')
    plt.close()


# 保存预测结果
def save_predictions(model, tokenizer, data1, data2):
    def predict(texts):
        dataset = TextDataset(texts, pd.Series([0] * len(texts)), tokenizer, MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return predictions, probabilities

    # 处理博文数据
    texts1 = data1['fenci'].dropna().reset_index(drop=True)
    preds1, probs1 = predict(texts1)
    data1['情感分类'] = label_encoder.inverse_transform(preds1)

    data1.to_excel('./BERT_PyTorch/博文预测结果.xlsx', index=False)

    # 处理评论数据
    texts2 = data2['fenci'].dropna().reset_index(drop=True)
    preds2, probs2 = predict(texts2)
    data2['情感分类'] = label_encoder.inverse_transform(preds2)
    data2.to_excel('./BERT_PyTorch/评论预测结果.xlsx', index=False)


# 情感分布可视化
def plot_emotion_distribution():
    df1 = pd.read_excel('./BERT_PyTorch/博文预测结果.xlsx')
    df2 = pd.read_excel('./BERT_PyTorch/评论预测结果.xlsx')
    combined = pd.concat([df1, df2])

    counts = combined['情感分类'].value_counts()
    labels = ['积极' if x == 1 else '消极' for x in counts.index]

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%',
            colors=['#66b3ff', '#ff9999'], startangle=90)
    plt.title('情感分布')
    plt.savefig('./BERT_PyTorch/emotion_distribution.png')
    plt.close()


if __name__ == '__main__':
    train_model()