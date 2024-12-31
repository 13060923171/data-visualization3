import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


# 读取数据
data = pd.read_csv('new_data.csv')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['fenci'], data['label'], test_size=0.2, random_state=42)

# 标签编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# 使用 CountVectorizer
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# 使用 TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 词嵌入参数
max_features = 10000
max_len = 100

# 词典构建
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['fenci'].values)
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
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1

def nb1():
    # 训练 CountVectorizer 特征的贝叶斯模型
    nb_count = MultinomialNB()
    nb_count.fit(X_train_count, y_train)
    y_pred_count = nb_count.predict(X_test_count)
    metrics_nb_count = evaluate_model(y_test, y_pred_count, 'Naive Bayes (CountVectorizer)')

    # 计算每个点的FPR和TPR，这里假设正类是第二类（索引为1）
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_count)

    # 计算PR曲线的数据点
    precision, recall, _ = precision_recall_curve(y_test, y_pred_count)

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
    plt.savefig('NB_Count_ROC曲线.png')
    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig('NB_Count_PR曲线.png')

    return metrics_nb_count

def nb2():
    # 训练 TfidfVectorizer 特征的贝叶斯模型
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, y_train)
    y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
    metrics_nb_tfidf = evaluate_model(y_test, y_pred_tfidf, 'Naive Bayes (TfidfVectorizer)')

    # 计算每个点的FPR和TPR，这里假设正类是第二类（索引为1）
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_tfidf)

    # 计算PR曲线的数据点
    precision, recall, _ = precision_recall_curve(y_test, y_pred_tfidf)

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
    plt.savefig('NB_TFIDF_ROC曲线.png')
    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig('NB_TFIDF_PR曲线.png')

    return metrics_nb_tfidf

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
    lstm_model.fit(X_train_seq, y_train, epochs=30, batch_size=16, validation_split=0.1, callbacks=[early_stopping])
    y_pred_lstm = (lstm_model.predict(X_test_seq) > 0.5).astype("int32")
    metrics_lstm = evaluate_model(y_test, y_pred_lstm, 'LSTM')

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
    plt.savefig('LSTM_ROC曲线.png')
    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig('LSTM_PR曲线.png')

    return metrics_lstm

def gru():
    # 构建 GRU 模型
    def build_gru_model():
        model = Sequential()
        model.add(Embedding(max_features, 128, input_length=max_len))
        model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    gru_model = build_gru_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    gru_model.fit(X_train_seq, y_train, epochs=30, batch_size=16, validation_split=0.1, callbacks=[early_stopping])
    y_pred_gru = (gru_model.predict(X_test_seq) > 0.5).astype("int32")
    metrics_gru = evaluate_model(y_test, y_pred_gru, 'GRU')


    # 计算每个点的FPR和TPR，这里假设正类是第二类（索引为1）
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_gru)

    # 计算PR曲线的数据点
    precision, recall, _ = precision_recall_curve(y_test, y_pred_gru)

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
    plt.savefig('GRU_ROC曲线.png')
    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig('GRU_PR曲线.png')

    return metrics_gru



if __name__ == '__main__':
    # nb1()
    nb2()
    # lstm()
    gru()

# # 将所有模型的指标保存到一个列表中
# all_metrics = [
#     metrics_nb_count,
#     metrics_nb_tfidf,
#     metrics_lstm,
#     metrics_gru
# ]
#
# # 将指标列表转换为 DataFrame
# metrics_df = pd.DataFrame(all_metrics)
#
# # 保存到 CSV 文件
# metrics_df.to_csv('model_metrics.csv', index=False)
# print("模型指标已保存到 model_metrics.csv 文件中")