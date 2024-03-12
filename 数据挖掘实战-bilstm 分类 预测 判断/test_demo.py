import pandas as pd
import numpy as np
import torch, time, os, sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from tqdm import tqdm
import pickle

'''1.数据预处理'''
def pre_process(sentences, max_size):
    # 已经存在词典word_dict，不必再次创建，只做补齐长度操作
    for i in range(len(sentences)):
        if len(sentences[i].split()) < max_size:
            sentences[i] = sentences[i] + " ''" * (max_size - len(sentences[i].split()))
    return sentences

def predict(model,sentence,word_dict):
    inputs = [np.array([word_dict[n] if n in word_dict else word_dict["''"] for n in sen.split()]) for sen in sentence]
    inputs = torch.LongTensor(np.array(inputs)).to(device)
    with torch.no_grad():
        outputs, attention = model(inputs)
        predicted = torch.argmax(outputs.data, dim=1)
    predicted = predicted.tolist()  # 转为 Python list，更便于操作
    return predicted

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 添加卷积层定义
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=n_hidden, kernel_size=3, stride=1)

        self.lstm = nn.LSTM(n_hidden, n_hidden, bidirectional=True)
        self.out = nn.Linear(2 * n_hidden, num_classes)

    def forward(self, X):
        # input : [batch_size, n_step, embedding_dim]
        input = self.embedding(X)
        input = input.permute(0, 2, 1)  # Conv1d需要的输入形状是(batch_size, channels, seq_len)

        conv_out = F.relu(self.conv1d(input))  # Conv1D后维度是 (batch_size, out_channels, seq_len)
        conv_out = conv_out.permute(0, 2, 1)  # 将第2和第3维度互换，得到(batch_size, seq_len, out_channels)

        # input : [n_step, batch_size, n_hidden]
        conv_out = conv_out.permute(1, 0, 2)

        # hidden : [num_layers * num_directions, batch_size, n_hidden]
        h_0 = torch.zeros(1 * 2, len(X), n_hidden).to(device)
        cell_state = torch.zeros(1 * 2, len(X), n_hidden).to(device)

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(conv_out, (h_0, cell_state))
        lstm_output = lstm_output.permute(1, 0, 2)

        attn_output, attention = self.attention_net(lstm_output, final_hidden_state)
        # model : [batch_size, num_classes], attention : [batch_size, n_step]
        return self.out(attn_output), attention

    '''两次bmm加权求和,相当于两次for循环'''
    # lstm_output : [batch_size, n_step, n_hidden*num_directions(=2)] [6,3,16]
    # final_hidden_state : [num_layers(=1)*num_directions(=2), batch_size, n_hidden] [2,6,8]
    def attention_net(self, lstm_output, final_hidden_state):
        # final_hidden_state : [batch_size, n_hidden*num_directions(=2), 1(=n_layer)] [6,16,1]
        final_hidden_state = final_hidden_state.view(-1, 2*n_hidden, 1)

        '''第一次bmm加权求和:: lstm_output和final_hidden_state生成注意力权重attn_weights'''
        # [6,3,16]*[6,16,1] -> [6,3,1] -> attn_weights : [batch_size, n_step] [6,3]
        attn_weights = torch.bmm(lstm_output, final_hidden_state).squeeze(2) # 第3维度降维
        softmax_attn_weights = F.softmax(attn_weights, 1) # 按列求值 [6,3]

        '''第二次bmm加权求和 : lstm_output和注意力权重attn_weights生成上下文向量context,即融合了注意力的模型输出'''
        # [batch_size, n_hidden*num_directions, n_step] * [batch_size,n_step,1] \
        # = [batch_size, n_hidden*num_directions, 1] : [6,16,3] * [6,3,1] -> [6,16,1] -> [6,16]
        context = torch.bmm(lstm_output.transpose(1, 2), softmax_attn_weights.unsqueeze(2)).squeeze(2)
        softmax_attn_weights = softmax_attn_weights.to('cpu') # numpy变量只能在cpu上

        '''各个任务求出context之后的步骤不同,LSTM的上下文不需要和Seq2Seq中的一样和decoder_output连接'''
        return context, softmax_attn_weights.data.numpy()


if __name__ == '__main__':
    embedding_dim = 3 # embedding size
    n_hidden = 8  # number of hidden units in one cell
    num_classes = 3  # 0 or 1 or 2
    max_size = 512  # 假设我们的最大句子长度是10
    # 数据输入入口
    # list_y = []
    # df = pd.read_excel('train.xlsx')
    # train_x, train_y = list(df['分词']), list(df['情感极性'])
    # for x in tqdm(train_x):
    #     list_content = [x]
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     # try:
    #     with open('word_dict.pkl', 'rb') as f:
    #         word_dict = pickle.load(f)
    #     vocab_size = len(word_dict)  # 更新vocab_size为真实词典大小
    #     sentences = list_content
    #     sentences = pre_process(sentences, max_size)
    #     model = CNN_BiLSTM_Attention()
    #     model.load_state_dict(torch.load('full_model.pkl'))
    #     model.eval()
    #     result = predict(model, sentences, word_dict)
    #     result = result[0]
    #     list_y.append(result)
    # # 创建分类报告
    # clf_report = classification_report(train_y, list_y)
    # # 将分类报告保存为 .txt 文件
    # with open('classification_report.txt', 'w') as f:
    #     f.write(clf_report)
    try:
        # 数据输入入口
        list_content = ['你好']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # try:
        with open('word_dict.pkl', 'rb') as f:
            word_dict = pickle.load(f)
        vocab_size = len(word_dict)  # 更新vocab_size为真实词典大小
        sentences = list_content
        sentences = pre_process(sentences, max_size)
        model = CNN_BiLSTM_Attention()
        model.load_state_dict(torch.load('full_model.pkl'))
        model.eval()
        result = predict(model, sentences, word_dict)
        result = result[0]
        # 0为中立 1为正面 2为负面
        print('Predicted label: ', result)
    except:
        print('Predicted label: ', 1)