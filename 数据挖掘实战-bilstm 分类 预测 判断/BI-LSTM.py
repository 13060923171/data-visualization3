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
    chars = 30 * '*'
    embedding_dim = 3 # embedding size
    n_hidden = 8  # number of hidden units in one cell
    num_classes = 3  # 0 or 1
    '''GPU比CPU慢的原因大致为:
    数据传输会有很大的开销,而GPU处理数据传输要比CPU慢,
    而GPU在矩阵计算上的优势在小规模神经网络中无法明显体现出来
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_excel('train.xlsx')
    train_x, train_y = list(df['分词']), list(df['情感极性'])
    sentences = train_x
    labels = train_y

    '''1.数据预处理'''
    sentence, word_list, word_dict, vocab_size, max_size = pre_process(sentences)
    inputs, targets = make_batch(sentence)

    '''2.构建模型'''
    model = CNN_BiLSTM_Attention()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    '''3.训练'''
    print('{}\nTrain\n{}'.format('*'*30, '*'*30))
    loss_record = []
    for epoch in tqdm(range(10000)):
        optimizer.zero_grad()
        output, attention = model(inputs)
        output = output.to(device)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        if loss >= 0.001: # 连续30轮loss小于0.01则提前结束训练
            loss_record = []
        else:
            loss_record.append(loss.item())
            if len(loss_record) == 30:
                torch.save(model.state_dict(), "full_model.pkl")
                with open('word_dict.pkl', 'wb') as f:
                    pickle.dump(word_dict, f)
                break

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'Loss = {:.6f}'.format(loss))
            torch.save(model.state_dict(), "full_model.pkl")
            # 保存词汇表
            with open('word_dict.pkl', 'wb') as f:
                pickle.dump(word_dict, f)

