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
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib
from joblib import dump, load
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

'''1.数据预处理'''
def pre_process(sentences, max_size):
    # 已经存在词典word_dict，不必再次创建，只做补齐长度操作
    for i in range(len(sentences)):
        if len(sentences[i].split()) < max_size:
            sentences[i] = sentences[i] + " ''" * (max_size - len(sentences[i].split()))
    return sentences

def predict(model, sentence, word_dict, device):
    # 确保输入数据格式正确，并转移到设定的设备上
    inputs = [np.array([word_dict[n] if n in word_dict else word_dict["''"] for n in sen.split()]) for sen in sentence]
    inputs = torch.LongTensor(np.array(inputs)).to(device)  # 直接转移到指定设备

    # 无梯度计算，适合推断模式
    with torch.no_grad():
        outputs, attention = model(inputs)
        predicted = torch.argmax(outputs.data, dim=1)

    # 转换为Python list用于返回和显示
    predicted = predicted.tolist()
    return predicted

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, num_classes):
        super(CNN_BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=n_hidden, kernel_size=3, stride=1)
        self.lstm = nn.LSTM(n_hidden, n_hidden, bidirectional=True)
        self.out = nn.Linear(2 * n_hidden, num_classes)
        self.dropout = nn.Dropout(0.9)

    def forward(self, X):
        # input : [batch_size, n_step, embedding_dim]
        input = self.embedding(X)
        input = input.permute(0, 2, 1)  # Conv1d需要的输入形状是(batch_size, channels, seq_len)

        conv_out = self.dropout(F.relu(self.conv1d(input)))
        conv_out = conv_out.permute(0, 2, 1)  # 将第2和第3维度互换，得到(batch_size, seq_len, out_channels)

        # input : [n_step, batch_size, n_hidden]
        conv_out = conv_out.permute(1, 0, 2)

        # hidden : [num_layers * num_directions, batch_size, n_hidden]
        h_0 = torch.zeros(1 * 2, len(X), n_hidden).to(device)
        cell_state = torch.zeros(1 * 2, len(X), n_hidden).to(device)

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(conv_out, (h_0, cell_state))
        lstm_output = lstm_output.permute(1, 0, 2)
        lstm_output = self.dropout(lstm_output)
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
    # 模型实例化
    embedding_dim = 3 # embedding size
    n_hidden = 6  # number of hidden units in one cell
    num_classes = 2  # 0 or 1
    max_size = 512
    # 数据输入入口
    df = pd.read_excel('../new_data.xlsx')
    df['content'] = df['content'].astype(str)
    df['type'] = df['type'].apply(lambda x: 1 if str(x) == 'pos' else 0)


    y_test = list(df['type'])
    list_content = list(df['content'])

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # try:
    with open('word_dict.pkl', 'rb') as f:
        word_dict = pickle.load(f)
    vocab_size = len(word_dict)  # 更新vocab_size为真实词典大小
    sentences = list_content
    sentences = pre_process(sentences, max_size)
    model = CNN_BiLSTM_Attention(vocab_size, embedding_dim, n_hidden, num_classes)
    model = model.to(device)  # 注意添加这行代码c
    model.load_state_dict(torch.load('full_model.pkl', map_location=device))
    model.eval()
    result = predict(model, sentences, word_dict,device)

    # 预测测试集，返回的是每个类别的概率
    y_pred_prob = result
    # 计算准确率、精确率、召回率和F1值
    clf_report = classification_report(y_test, y_pred_prob)
    # 将分类报告保存为 .txt 文件
    with open('classification_report.txt', 'w') as f:
        f.write(clf_report)
    # 计算每个点的FPR和TPR，这里假设正类是第二类（索引为1）
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # 计算PR曲线的数据点
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

    # 计算AUC值
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('cnn_ROC曲线.png')
    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig('cnn_PR曲线.png')

    df1 = pd.read_csv('cnn_bi_lstm loss accuracy_data.csv',encoding='utf-8-sig')
    # 可视化训练历史
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(list(df1['Epoch']),list(df1['train_loss']), label='Train loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(list(df1['Epoch']),list(df1['train_acc']), label='Train accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('Model loss accuracy.png')
    plt.show()

    df['情感type'] = result


    def demo2(x):
        x1 = int(x)
        if x1 == 1:
            return 'pos'
        else:
            return 'neg'


    df['情感type'] = df['情感type'].apply(demo2)

    new_df = df['情感type'].value_counts()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(16, 9), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('情感占比分布')
    plt.tight_layout()
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.savefig('情感占比分布.png')
    df.to_excel('new_cnn_bi_lst.xlsx', index=False)