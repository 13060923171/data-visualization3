import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# 加载预训练的中文BERT模型及其分词器
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 从csv文件中读取数据
df = pd.read_excel('train_data.xlsx')
df = df.dropna(subset=['fenci'],axis=0)
df['fenci'] = df['fenci'].astype('str')

def tidai(x):
    x1 = str(x)
    if x1 == '-1':
        return 2
    elif x1 == '11':
        return 1
    elif x1 == '-11':
        return 2
    else:
        return x1


df['情感分类'] = df['情感分类'].apply(tidai)
df['情感分类'] = df['情感分类'].astype('int')

# 准备数据集（假设train_texts和train_labels是训练集）
train_texts = list(df['fenci'])
train_labels =list(df['情感分类'])
# 训练模型
optimizer = AdamW(model.parameters(), lr=1e-5)
epochs = 5
accuracies = []

for epoch in tqdm(range(epochs)):
    for text, label in zip(train_texts, train_labels):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs, labels=torch.tensor([label]))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 在每个epoch结束时计算准确率
    predictions = []
    for text, label in zip(train_texts, train_labels):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        predictions.append(predicted_label)

    accuracy = accuracy_score(train_labels, predictions)
    accuracies.append(accuracy)
    print(f"Epoch {epoch + 1}, Accuracy: {accuracy}")

# 生成准确率的折线图
plt.plot(range(1, epochs + 1), accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.savefig('Training Accuracy.png')
# plt.show()

# 保存模型（这里仅为示例，实际使用时应保存模型权重及结构）
torch.save(model.state_dict(), 'bert_model.pth')

# 使用训练好的模型进行分类预测
# 加载模型
model.load_state_dict(torch.load('bert_model.pth'))
model.eval()

data_val = pd.read_excel('test_data.xlsx')
data_val = data_val.dropna(subset=['fenci'],axis=0)
# 准备新的文本数据进行分类预测
new_text = list(data_val['fenci'])
predicted_labels = []
for text in tqdm(new_text):
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        predicted_labels.append(predicted_label)
    except:
        predicted_labels.append(" ")

data_val['情感分类'] = predicted_labels
def tidai2(x):
    x1 = str(x)
    if x1 == '2':
        return -1
    else:
        return x1


data_val['情感分类'] = data_val['情感分类'].apply(tidai2)
data_val['情感分类'] = data_val['情感分类'].astype('str')
data_val.to_excel('new_test_data.xlsx')
