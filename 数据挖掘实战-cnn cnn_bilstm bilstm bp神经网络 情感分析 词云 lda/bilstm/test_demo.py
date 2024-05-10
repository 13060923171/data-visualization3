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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # try:
    with open('word_dict.pkl', 'rb') as f:
        word_dict = pickle.load(f)
    vocab_size = len(word_dict)  # 更新vocab_size为真实词典大小
    sentences = list_content
    sentences = pre_process(sentences, max_size)
    model = BiLSTM_Attention(vocab_size, embedding_dim, n_hidden, num_classes)
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
    plt.savefig('bi_lstm_ROC曲线.png')
    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig('bi_lstm_PR曲线.png')

    df1 = pd.read_csv('bi_lstm loss accuracy_data.csv',encoding='utf-8-sig')
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
    df.to_excel('new_bi_lstm.xlsx', index=False)