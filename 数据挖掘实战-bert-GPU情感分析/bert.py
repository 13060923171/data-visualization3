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
        #将类别标签转化为数字
        self.labels = [label for label in df['评论类型']]
        #对每个文本进行分词并进行最大长度填充和截断
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length =512,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['评价内容']]
    #返回所有的标签
    def classes(self):
        return self.labels
    #返回数据集大小
    def __len__(self):
        return len(self.labels)
    #获取批次的标签
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
    # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    #val_dataloader是一个Pytorch的数据加载器对象，它按批次自动加载预处理后的数据，并返回包含多个输入文本和标签的元组。
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=32)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    # 将模型移到GPU上
    device = torch.device('cuda')
    model.to(device=device)
    val_acc_list = []  # 用于存储每个 epoch 结束后在验证集上的准确率
    x_data = []
    # 开始进入训练循环
    for epoch_num in tqdm(range(epochs)):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in train_dataloader:
            # 将标签转移到cpu设备上
            train_label = train_label.to(device=device)
            # 将attention_mask转移到GPU设备上
            mask = train_input['attention_mask'].to(device=device)
            # 将input_ids转移到GPU设备上，并去除从train_input的第一维生成的空维度
            input_id = train_input['input_ids'].squeeze(1).to(device=device)

            # 通过模型得到输出
            output = model(input_id, mask)
            train_label = train_label.type(torch.long)
            # 计算损失
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            torch.cuda.empty_cache()
        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.to(device=device)
                mask = val_input['attention_mask'].to(device=device)
                input_id = val_input['input_ids'].squeeze(1).to(device=device)

                output = model(input_id, mask)
                val_label = val_label.type(torch.long)
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
                # 计算验证集的准确率
            val_acc = total_acc_val / len(val_dataloader.dataset)  # 计算验证集上的准确率
            # val_acc = total_acc_val / len(val_data)  # 计算验证集上的准确率
            val_acc = round(val_acc,3)
            val_acc_list.append(val_acc)  # 将准确率添加到列表中
        x_data.append(epoch_num + 1)
        torch.cuda.empty_cache()
    #保存训练好的模型
    # torch.save(model.state_dict(), "full_model.pkl")

    return val_acc_list,x_data  # 返回验证集准确率列表


if __name__ == '__main__':
    # 从csv文件中读取数据
    df = pd.read_excel('train.xlsx').iloc[:50]
    df = df.dropna(subset=['评价内容'],axis=0)
    df['评价内容'] = df['评价内容'].astype('str')
    df['评论类型'] = df['评论类型'].astype('int')
    # 将数据集划分为训练集、验证集、测试集
    ## random_state=42 表示设定随机种子，以确保结果可复现
    data_train, data_test = train_test_split(df, test_size=0.3, random_state=42)

    data_val = pd.read_excel('test.xlsx')
    EPOCHS = 6
    model = BertClassifier()
    LR = 0.001

    val_acc_list,x_data = train(model, data_train, data_test, LR, EPOCHS)
    #绘制准确率折线图
    plt.plot(x_data, val_acc_list,'^--',color='#2614e8',label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trend')
    plt.savefig('Accuracy Trend.png')

    df22 = pd.DataFrame()
    df22['Epoch'] = x_data
    df22['Accuracy'] = val_acc_list
    df22.to_excel('Accuracy.xlsx')
