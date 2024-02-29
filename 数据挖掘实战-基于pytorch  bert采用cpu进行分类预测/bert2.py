import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import random
import pandas as pd

# 数据集定义
class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        inputs = {key: val.squeeze() for key, val in encoding.items()}
        inputs['labels'] = torch.tensor(label)

        return inputs

df = pd.read_excel('train_data.xlsx').iloc[:50]
def tidai(x):
    x1 = str(x)
    if x1 == '-1':
        return 2
    else:
        return x1

df['情感分类'] = df['情感分类'].apply(tidai)
df['情感分类'] = df['情感分类'].astype('int')
texts = list(df['post_title'])
labels = list(df['情感分类'])


# 数据集拆分
def split_dataset(texts, labels, split_ratio=0.75):
    data = list(zip(texts, labels))
    random.shuffle(data)
    n_train = int(len(data) * split_ratio)
    train_data = data[:n_train]
    test_data = data[n_train:]
    train_texts, train_labels = zip(*train_data)
    test_texts, test_labels = zip(*test_data)
    return train_texts, train_labels, test_texts, test_labels


train_texts, train_labels, test_texts, test_labels = split_dataset(texts, labels)

# 加载预训练的 BERT 模型和 tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 创建训练集和测试集的数据集和数据加载器
train_dataset = MyDataset(train_texts, train_labels, tokenizer)
test_dataset = MyDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 训练和优化
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

accuracies = []  # 保存准确率
x_data = []
model.train()
for epoch in range(1):
    for batch in train_loader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 在测试集上评估准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            _, predicted_labels = torch.max(logits, 1)
            total += inputs['labels'].size(0)
            correct += (predicted_labels == inputs['labels']).sum().item()

    accuracy = correct / total
    accuracies.append(accuracy)
    print(f"Epoch {epoch + 1} Accuracy: {accuracy}")
    x_data.append(epoch + 1)
    torch.save(model, "bert_model.pkl")

# 绘制准确率折线图
plt.plot(range(1, len(accuracies) + 1), accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Trend')
plt.savefig('Accuracy Trend.png')
plt.show()

df22 = pd.DataFrame()
df22['Epoch'] = x_data
df22['Accuracy'] = accuracies
df22.to_excel('Accuracy.xlsx')

data_val = pd.read_excel('test_data.xlsx').iloc[:50]
# 使用模型进行新数据预测
new_data = list(data_val['post_title'])
new_dataset = MyDataset(new_data, [0, 0], tokenizer)
new_loader = DataLoader(new_dataset, batch_size=1, shuffle=False)

model.eval()
all_predictions = []
with torch.no_grad():
    for batch in new_loader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, 1)
        all_predictions.append(predicted_labels.item())
        # for i, text in enumerate(batch['input_ids']):
        #     print(f"文本: {tokenizer.decode(text)}，预测分类: {predicted_labels[i]}")
        #     all_predictions.append(predicted_labels[i])

data_val['情感分类'] = all_predictions
def tidai2(x):
    x1 = str(x)
    if x1 == '2':
        return -1
    else:
        return x1

data_val['情感分类'] = data_val['情感分类'].apply(tidai2)
data_val['情感分类'] = data_val['情感分类'].astype('str')
data_val.to_excel('new_test_data.xlsx')