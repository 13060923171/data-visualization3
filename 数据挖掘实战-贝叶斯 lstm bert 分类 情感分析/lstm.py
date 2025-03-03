import pandas as pd
import numpy as np
import jieba
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


# 读取数据
data1 = pd.read_csv('./train/新_博文表.csv')
data2 = pd.read_csv('./train/新_评论表.csv')

df1 = pd.DataFrame()
df1['fenci'] = data1['fenci']
df1['label'] = data1['label']

df2 = pd.DataFrame()
df2['fenci'] = data2['fenci']
df2['label'] = data2['label']

df = pd.concat([df1,df2],axis=0)
if not os.path.exists("./LSTM"):
    os.mkdir("./LSTM")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['fenci'], df['label'], test_size=0.3, random_state=42)

# 标签编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# 词嵌入参数
max_features = 10000
max_len = 100

# 词典构建
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['fenci'].values)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# 序列填充
X_train_seq = pad_sequences(X_train_seq, maxlen=max_len)
X_test_seq = pad_sequences(X_test_seq, maxlen=max_len)


# 评估函数
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 创建指标数据
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [accuracy, precision, recall, f1]

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#ccd5ae', '#e9edc9', '#fefae0', '#faedcd'])
    plt.ylim(0, 1.1)
    plt.title(f'Model Performance Metrics: {model_name}')
    plt.ylabel('Score')

    # 在柱子上方显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.savefig(f'./LSTM/{model_name}_性能指标.png')
    plt.close()  # 关闭图形防止内存泄漏

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1

def lstm():
    # 构建 LSTM 模型
    def build_lstm_model():
        model = Sequential()
        model.add(Embedding(max_features, 128, input_length=max_len))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    lstm_model = build_lstm_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 训练模型并保存训练历史
    history = lstm_model.fit(
        X_train_seq,
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练损失', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_loss'], label='验证损失', color='#ff7f0e', linestyle='--', linewidth=2)

    plt.title('LSTM 训练损失曲线', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 标记最佳epoch（EarlyStopping恢复的权重）
    best_epoch = early_stopping.stopped_epoch - early_stopping.patience + 1
    if best_epoch >= 0:
        plt.axvline(best_epoch, color='red', linestyle=':', linewidth=1,
                    label=f'最佳epoch ({best_epoch + 1})')
        plt.legend()

    plt.tight_layout()
    plt.savefig('./LSTM/LSTM训练损失曲线.png')
    plt.close()

    y_pred_lstm = (lstm_model.predict(X_test_seq) > 0.5).astype("int32")
    metrics_lstm = evaluate_model(y_test, y_pred_lstm, 'LSTM')

    # 新增预测保存函数
    def save_lstm_predictions(data, filename):
        # 转换特征
        seq = tokenizer.texts_to_sequences(data['fenci'])
        padded_seq = pad_sequences(seq, maxlen=max_len)

        # 预测数值标签
        y_pred = lstm_model.predict(padded_seq)
        y_pred_numeric = (y_pred > 0.5).astype(int).flatten()

        # 转换回原始标签
        data['情感分类'] = label_encoder.inverse_transform(y_pred_numeric)

        data = data.drop(['label', 'score'], axis=1)
        # 保存结果
        data.to_excel(f'./LSTM/predictions_{filename}', index=False)
        print(f"预测结果已保存至 predictions_{filename}")

    # 处理原始数据
    save_lstm_predictions(data1, '新_博文表.xlsx')
    save_lstm_predictions(data2, '新_评论表.xlsx')


    def emotion_pie(label):
        d = {}
        for l in label:
            d[l] = d.get(l, 0) + 1
        x_data = []
        y_data = []
        for x,y in d.items():
            x_data.append(x)
            y_data.append(y)
        plt.figure(figsize=(9, 6), dpi=500)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
        plt.title(f'情感分布情况')
        plt.tight_layout()
        # 添加图例
        plt.legend(x_data, loc='lower right')
        plt.savefig(f'./LSTM/情感分布情况.png')


    new_df1 = pd.read_excel('./LSTM/predictions_新_博文表.xlsx')
    new_df2 = pd.read_excel('./LSTM/predictions_新_评论表.xlsx')
    label1 = new_df1['情感分类'].tolist()
    label2 = new_df2['情感分类'].tolist()
    label3 = label1 + label2

    emotion_pie(label3)


    # 计算每个点的FPR和TPR，这里假设正类是第二类（索引为1）
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_lstm)

    # 计算PR曲线的数据点
    precision, recall, _ = precision_recall_curve(y_test, y_pred_lstm)

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
    plt.savefig('./LSTM/lstm_ROC曲线.png')
    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig('./LSTM/lstm_PR曲线.png')


if __name__ == '__main__':
    lstm()

