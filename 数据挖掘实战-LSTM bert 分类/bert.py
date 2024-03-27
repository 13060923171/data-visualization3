# 导入必要的库
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from transformers import BertModel
from sklearn.metrics import f1_score
import torch
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


## 导入BertTokenizer,做分词处理
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        #将类别是否是涉警舆情转化为数字
        self.labels = [label for label in df['是否是涉警舆情']]
        #对每个文本进行分词并进行最大长度填充和截断
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length =512,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['fenci']]
    #返回所有的是否是涉警舆情
    def classes(self):
        return self.labels
    #返回数据集大小
    def __len__(self):
        return len(self.labels)
    #获取批次的是否是涉警舆情
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])
    #获取批次的输入文本
    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    #获取指定索引的案例
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


#定义bert分类器模型
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        #加载bert模型
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        #定义dropout
        self.dropout = nn.Dropout(dropout)
        #定义线性层
        self.linear = nn.Linear(768, 5)
        #定义relu激活函数
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        #进行bert模型的正向传播计算
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        #进行dropout层的正向传播计算
        dropout_output = self.dropout(pooled_output)
        #进行线性层传播计算
        linear_output = self.linear(dropout_output)
        #进行relu激活函数的正向传播计算
        final_layer = self.relu(linear_output)
        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)

    x_data = []

    train_loss_list = []  # 添加用于存储训练集损失的列表
    train_acc_list = []  # 添加用于存储训练集准确率的列表
    val_loss_list = []  # 添加用于存储验证集损失的列表
    val_acc_list = []  # 添加用于存储验证集准确率的列表

    for epoch_num in tqdm(range(epochs)):
        total_acc_train = 0
        total_loss_train = 0
        for train_input, train_label in train_dataloader:
            train_label = train_label.to(device=device)
            mask = train_input['attention_mask'].to(device=device)
            input_id = train_input['input_ids'].squeeze(1).to(device=device)
            output = model(input_id, mask)
            train_label = train_label.type(torch.long)
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            model.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            torch.cuda.empty_cache()
        train_loss_list.append(total_loss_train / len(train_data))  # 平均训练损失
        train_acc_list.append(total_acc_train / len(train_data))  # 平均训练准确率

        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device=device)
                mask = val_input['attention_mask'].to(device=device)
                input_id = val_input['input_ids'].squeeze(1).to(device=device)
                output = model(input_id, mask)
                val_label = val_label.type(torch.long)
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        val_loss_list.append(total_loss_val / len(val_data))  # 平均验证损失
        val_acc_list.append(total_acc_val / len(val_data))  # 平均验证准确率
        x_data.append(epoch_num + 1)
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), "full_model.pkl")


    # 返回数据，便于后续可视化
    return x_data,train_loss_list, train_acc_list, val_loss_list, val_acc_list


if __name__ == '__main__':
    # 从csv文件中读取数据
    df = pd.read_excel('train.xlsx')
    df = df.dropna(subset=['fenci'],axis=0)
    df['fenci'] = df['fenci'].astype('str')
    df['是否是涉警舆情'] = df['是否是涉警舆情'].astype('int')
    # 将数据集划分为训练集、验证集、测试集
    ## random_state=42 表示设定随机种子，以确保结果可复现
    data_train, data_test = train_test_split(df, test_size=0.3, random_state=42)
    EPOCHS = 10
    model = BertClassifier()
    LR = 1e-6
    x_data,train_loss_list, train_acc_list, val_loss_list, val_acc_list = train(model, data_train, data_test, LR, EPOCHS)

    # 可视化训练历史
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(x_data,train_loss_list, label='Train loss')
    axes[0].plot(x_data,val_loss_list, label='Validation loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(x_data,train_acc_list, label='Train accuracy')
    axes[1].plot(x_data,val_acc_list, label='Validation accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('bert_Model loss accuracy.png')
    plt.show()

    data = pd.DataFrame()
    data['Epoch'] = x_data
    data['train_loss'] = train_loss_list
    data['train_acc'] = train_acc_list
    data['val_loss'] = val_loss_list
    data['val_acc'] = val_acc_list
    data.to_excel('bert_Model loss accuracy_data.xlsx',index=False)