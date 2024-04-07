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
from sklearn.metrics import recall_score, f1_score
import copy

## 导入BertTokenizer,做分词处理
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        #将类别fenci转化为数字
        self.labels = [label for label in df['情感分类']]
        #对每个文本进行分词并进行最大长度填充和截断
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length =512,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['fenci']]
    #返回所有的fenci
    def classes(self):
        return self.labels
    #返回数据集大小
    def __len__(self):
        return len(self.labels)
    #获取批次的fenci
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


def train(model, train_data, val_data, learning_rate, epochs, patience):
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)

    x_data = []
    best_model = None  # 用于存储最佳模型的状态
    early_stopping_counter = 0
    min_loss_val = np.inf
    train_loss_list = []  # 添加用于存储训练集损失的列表
    train_acc_list = []  # 添加用于存储训练集准确率的列表
    val_loss_list = []  # 添加用于存储验证集损失的列表
    val_acc_list = []  # 添加用于存储验证集准确率的列表
    train_recall_list = []
    train_f1_list = []
    val_recall_list = []
    val_f1_list = []
    for epoch_num in tqdm(range(epochs)):
        total_acc_train = 0
        total_loss_train = 0
        # 新增用于保存预测值和标签的列表
        pred_label_list = []
        true_label_list = []
        for train_input, train_label in train_dataloader:
            train_label = train_label.to(device=device)
            mask = train_input['attention_mask'].to(device=device)
            input_id = train_input['input_ids'].squeeze(1).to(device=device)
            output = model(input_id, mask)
            train_label = train_label.type(torch.long)
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            pred_label = output.argmax(dim=1)
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            # 将此批次的预测值和标签添加到相应的列表中
            pred_label_list.extend(pred_label.tolist())
            true_label_list.extend(train_label.tolist())

            model.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            torch.cuda.empty_cache()

        # 在每个周期结束时，计算召回率和F1分数
        recall = recall_score(true_label_list, pred_label_list, average='macro')
        f1 = f1_score(true_label_list, pred_label_list, average='macro')

        train_loss_list.append(total_loss_train / len(train_data))  # 平均训练损失
        train_acc_list.append(total_acc_train / len(train_data))  # 平均训练准确率
        print('epoch:',epoch_num)
        print('train_loss:',round(total_loss_train / len(train_data),4))
        print('train_acc:', round(total_acc_train / len(train_data), 4))
        print('recall:', round(recall,4))
        print('f1 score:', round(f1,4))
        # 将这一周期的召回率和F1分数添加到相应的列表中
        train_recall_list.append(recall)  # 平均训练召回率
        train_f1_list.append(f1)  # 平均训练F1分数

        total_acc_val = 0
        total_loss_val = 0
        pred_label_val = []
        true_label_val = []
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device=device)
                mask = val_input['attention_mask'].to(device=device)
                input_id = val_input['input_ids'].squeeze(1).to(device=device)
                output = model(input_id, mask)
                val_label = val_label.type(torch.long)
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                pred_label = output.argmax(dim=1)
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

                # 将此批次的预测值和标签添加到相应的列表中
                pred_label_val.extend(pred_label.tolist())
                true_label_val.extend(val_label.tolist())

        # 在每个周期结束时，计算召回率和F1分数
        recall = recall_score(true_label_val, pred_label_val, average='macro')
        f1 = f1_score(true_label_val, pred_label_val, average='macro')
        val_loss_list.append(total_loss_val / len(val_data))  # 平均验证损失
        val_acc_list.append(total_acc_val / len(val_data))  # 平均验证准确率
        # 将这一周期的召回率和F1分数添加到相应的列表中
        val_recall_list.append(recall)  # 平均训练召回率
        val_f1_list.append(f1)  # 平均训练F1分数
        x_data.append(epoch_num + 1)
        torch.cuda.empty_cache()

        if total_loss_val < min_loss_val:
            min_loss_val = total_loss_val
            early_stopping_counter = 0  # reset early stopping counter
            best_model = copy.deepcopy(model.state_dict())  # 更新最佳模型状态
        else:
            early_stopping_counter += 1  # increase early stopping counter

        if early_stopping_counter >= patience:
            print("Early stopping!")
            break

    if best_model is not None:  # 如果最佳模型存在，我们保存它
        torch.save(best_model, "best_model.pkl")
    else:  # 否则，可能所有的模型都没有得到训练，我们就保存最后的模型
        torch.save(model.state_dict(), "full_model.pkl")


    # 返回数据，便于后续可视化
    return x_data,train_loss_list, train_acc_list,train_recall_list,train_f1_list, val_loss_list, val_acc_list,val_recall_list,val_f1_list


if __name__ == '__main__':
    # 从csv文件中读取数据
    # 从csv文件中读取数据
    df = pd.read_excel('train_data.xlsx').iloc[:30]
    df = df.dropna(subset=['fenci'], axis=0)
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
    # 将数据集划分为训练集、验证集、测试集
    ## random_state=42 表示设定随机种子，以确保结果可复现
    data_train, data_test = train_test_split(df, test_size=0.3, random_state=42)
    EPOCHS = 10
    model = BertClassifier()
    LR = 1e-6
    patience = 5
    x_data,train_loss_list, train_acc_list,train_recall_list,train_f1_list, val_loss_list, val_acc_list,val_recall_list,val_f1_list = train(model, data_train, data_test, LR, EPOCHS, patience)

    # 可视化训练历史
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(x_data,train_loss_list, label='Train loss')
    axes[0, 0].plot(x_data,val_loss_list, label='Validation loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(x_data,train_acc_list, label='Train accuracy')
    axes[0, 1].plot(x_data,val_acc_list, label='Validation accuracy')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    axes[1, 0].plot(x_data, train_recall_list, label='Train recall')
    axes[1, 0].plot(x_data, val_recall_list, label='Validation recall')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('recall')
    axes[1, 0].legend()

    axes[1, 1].plot(x_data, train_f1_list, label='Train F1 Score')
    axes[1, 1].plot(x_data, val_f1_list, label='Validation F1 Score')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('bert_Model loss accuracy.png')
    plt.show()

    data = pd.DataFrame()
    data['Epoch'] = x_data
    data['train_loss'] = train_loss_list
    data['train_acc'] = train_acc_list
    data['train_recall'] = train_recall_list
    data['train_f1'] = train_f1_list
    data['val_loss'] = val_loss_list
    data['val_acc'] = val_acc_list
    data['val_recall'] = val_recall_list
    data['val_f1'] =  val_f1_list

    data.to_excel('bert_Model loss accuracy_data.xlsx',index=False)