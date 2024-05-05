import pandas as pd
import numpy as np
import torch, time, os, sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from tqdm import tqdm
import pickle

# 全局变量
word_dict = {}

'''1.数据预处理'''
def pre_process(sentences):
    global word_dict  # 使用全局变量
    word_sequence = " ".join(sentences).split()
    word_list = []
    '''
    如果用list(set(word_sequence))来去重,得到的将是一个随机顺序的列表(因为set无序),
    这样得到的字典不同,保存的上一次训练的模型很有可能在这一次不能用
    '''
    for word in word_sequence:
        if word not in word_list:
            word_list.append(word)
    word_dict = {w:i for i, w in enumerate(word_list)}
    word_dict["''"] = len(word_dict)
    word_list = word_list.append("''")
    vocab_size = len(word_dict) # 词库大小16
    max_size = 0
    for sen in sentences:
        if len(sen.split()) > max_size:
            max_size = len(sen.split()) # 最大长度3
    for i in range(len(sentences)):
        if len(sentences[i].split()) < max_size:
            sentences[i] = sentences[i] + " ''" * (max_size - len(sentences[i].split()))
    # 在预处理完成后，保存词汇表
    with open('word_dict.pkl', 'wb') as f:
        pickle.dump(word_dict, f)
    return sentences, word_list, word_dict, vocab_size, max_size


def make_batch(sentences):
    # 对于每个句子,返回包含句子内每个单词序号的列表
    inputs = [np.array([word_dict[n] for n in sen.split()]) for sen in sentences] # [6,3]
    targets = [out for out in labels]
    inputs = torch.LongTensor(np.array(inputs)).to(device)
    targets = torch.LongTensor(np.array(targets)).to(device)
    '''情感分类构建嵌入矩阵,没有eye()'''
    return inputs, targets


class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(2 * n_hidden, num_classes)
        self.dropout = nn.Dropout(0.9)

    def forward(self, X):
        # X : [batch_size, max_len]
        embedded = self.embedding(X)  # output shape: [batch_size, max_len, embedding_dim]

        # LSTM input: [max_len, batch_size, embedding_dim]
        lstm_input = embedded.permute(1, 0, 2)

        # Initialize hidden and cell states
        # Dimensions: [num_layers * num_directions, batch_size, n_hidden]
        h_0 = torch.zeros(1 * 2, X.size(0), n_hidden).to(lstm_input.device)
        c_0 = torch.zeros(1 * 2, X.size(0), n_hidden).to(lstm_input.device)

        # lstm_output: [max_len, batch_size, n_hidden * num_directions]
        # final_hidden_state: [num_layers * num_directions, batch_size, n_hidden]
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(lstm_input, (h_0, c_0))

        # LSTM output to attention layer
        # first, concatenate the hidden states of both directions
        final_hidden = torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:]), dim=1)
        attn_output, attention = self.attention_net(lstm_output, final_hidden)

        # Apply Dropout before the output layer
        out_with_dropout = self.dropout(attn_output)

        # Pass the output through the linear layer
        output = self.out(out_with_dropout)

        return output, attention

    def attention_net(self, lstm_output, final_hidden_state):
        # lstm_output : shape [max_len, batch_size, n_hidden * 2]
        # final_hidden_state : shape [batch_size, n_hidden * 2]
        hidden = final_hidden_state.unsqueeze(2)    # shape [batch_size, n_hidden * 2, 1]

        # attn_weights : shape [batch_size, max_len]
        attn_weights = torch.bmm(lstm_output.permute(1, 0, 2), hidden).squeeze(2)
        softmax_attn_weights = F.softmax(attn_weights, dim=1)

        # context: shape [batch_size, n_hidden * 2]
        context = torch.bmm(lstm_output.permute(1, 2, 0), softmax_attn_weights.unsqueeze(2)).squeeze(2)
        return context, softmax_attn_weights.data.cpu().numpy()


if __name__ == '__main__':
    embedding_dim = 3 # embedding size
    n_hidden = 6  # number of hidden units in one cell
    num_classes = 2  # 0 or 1
    '''GPU比CPU慢的原因大致为:
    数据传输会有很大的开销,而GPU处理数据传输要比CPU慢,
    而GPU在矩阵计算上的优势在小规模神经网络中无法明显体现出来
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    df = pd.read_excel('../new_data.xlsx')
    df['content'] = df['content'].astype(str)
    df['type'] = df['type'].apply(lambda x: 1 if str(x) == 'pos' else 0)
    sentences, X_test, labels, y_test = train_test_split(list(df['content']), list(df['type']), test_size=0.2, random_state=42)

    data = pd.DataFrame()
    data['Epoch'] = ['Epoch']
    data['train_loss'] = ['train_loss']
    data['train_acc'] = ['train_acc']
    data.to_csv('bi_lstm loss accuracy_data.csv', index=False,header=False,mode='w',encoding='utf-8-sig')
    '''1.数据预处理'''
    sentence, word_list, word_dict, vocab_size, max_size = pre_process(sentences)
    inputs, targets = make_batch(sentence)

    '''2.构建模型'''
    model = BiLSTM_Attention(vocab_size, embedding_dim, n_hidden, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    '''3.训练'''
    loss_record = []
    for epoch in tqdm(range(10000)):
        total_train = 0
        correct_train = 0
        train_loss = 0
        optimizer.zero_grad()
        output, attention = model(inputs)
        output = output.to(device)
        loss = criterion(output, targets)
        train_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total_train += targets.size(0)  # 更新样本总数
        correct_train += (predicted == targets.to(device)).sum().item()  # 更新正确预测的数量

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            epoch = '%04d' % ((epoch + 1) /1000)
            train_acc = '{:.6f}'.format(correct_train / total_train)  # 计算准确率
            train_loss = '{:.6f}'.format(train_loss)  # 计算准确率
            data = pd.DataFrame()
            data['Epoch'] = [epoch]
            data['train_loss'] = [train_loss]
            data['train_acc'] = [train_acc]  # 存储计算出的准确率
            data.to_csv('bi_lstm loss accuracy_data.csv', index=False, header=False, mode='a+', encoding='utf-8-sig')
            torch.save(model.state_dict(), "full_model.pkl")
            # 保存词汇表
            with open('word_dict.pkl', 'wb') as f:
                pickle.dump(word_dict, f)

