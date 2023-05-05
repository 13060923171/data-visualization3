import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
import jieba
from sklearn.metrics import accuracy_score
from sklearn import svm
from tqdm import tqdm
import joblib
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
import numpy as np
df = pd.read_csv('./data/data_情感分析.csv')
new_df = df.dropna(subset=['分词'],axis=0)
stop_words = []

with open("stopwords_cn.txt", 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


def main1():
    # 创建LabelEncoder对象
    le = LabelEncoder()
    # 使用LabelEncoder对象对分类变量进行编码
    y = le.fit_transform(new_df['search-area'])

    # 将分词后的中文文本转换为数字类型
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(new_df['分词'])

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

    # best_k = score_list.index(max(score_list)) + 1
    # #406
    # print('Best K:', best_k)

    # 使用最优的K值训练模型并测试
    knn = KNeighborsClassifier(n_neighbors=406)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Knn Accuracy:', accuracy)

    # 训练和预测 - Naive Bayes
    clf_nb = MultinomialNB()
    clf_nb.fit(X_train, y_train)
    y_pred_nb = clf_nb.predict(X_test)
    print('Naive Bayes accuracy:', accuracy_score(y_test, y_pred_nb))

    # # 训练和预测 - SVM
    # clf_svm = SVC(kernel='linear')
    # clf_svm.fit(X_train, y_train)
    # y_pred_svm = clf_svm.predict(X_test)
    # print('SVM accuracy:', accuracy_score(y_test, y_pred_svm))


    # score_list = []
    # for i in tqdm(range(400, 410)):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(X_train, y_train)
    #     score = knn.score(X_test, y_test)
    #     score_list.append(round(score,4))

    # score_list1 = []
    # for i in tqdm(np.arange(0.1, 1.1, 0.1)):
    #     svm = SVC(C=i)
    #     svm.fit(X_train, y_train)
    #     score = svm.score(X_test, y_test)
    #     score_list1.append(score)


    # df = pd.DataFrame()
    # df['分值1'] = score_list
    # df['分值2'] = score_list1
    # df.to_csv('data.csv',encoding='utf-8-sig')
    # print(score_list)
    df = pd.read_csv('data.csv')
    x_data1 = list(df['分值1'])
    x_data2 = list(df['分值2'])
    y_data1 = [y for y in range(400, 410)]
    y_data2 = [y for y in np.arange(0.1, 1.1, 0.1)]
    fig = plt.figure(figsize=(20,12),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax1.plot(y_data1,x_data1, label='KNN',color='r')
    ax2.plot(y_data2,x_data2, label='SVM', color='g')
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('n_neighbors')
    ax1.set_ylabel('准确率')
    ax2.set_xlabel('惩罚系数')
    plt.title('准确率曲线图')
    plt.savefig('准确率曲线图.png')
    plt.show()


    # accuracy1 = round(metrics.accuracy_score(y_test, y_pred), 4)
    # accuracy2 = round(metrics.accuracy_score(y_test, y_pred_nb), 4)
    # accuracy3 = round(metrics.accuracy_score(y_test, y_pred_svm), 4)
    #
    # precision1 = round(metrics.precision_score(y_test, y_pred, average='micro'), 4)
    # precision2 = round(metrics.precision_score(y_test, y_pred_nb, average='micro'), 4)
    # precision3 = round(metrics.precision_score(y_test, y_pred_svm, average='micro'), 4)
    #
    # recall1 = round(metrics.recall_score(y_test, y_pred, average='micro'), 4)
    # recall2 = round(metrics.recall_score(y_test, y_pred_nb, average='micro'), 4)
    # recall3 = round(metrics.recall_score(y_test, y_pred_svm, average='micro'), 4)
    #
    # f1_1 = round(metrics.f1_score(y_test, y_pred, average='weighted'), 4)
    # f1_2 = round(metrics.f1_score(y_test, y_pred_nb, average='weighted'), 4)
    # f1_3 = round(metrics.f1_score(y_test, y_pred_svm, average='weighted'), 4)
    #
    #
    # data = pd.DataFrame()
    # data['准确率'] = ['KNN准确率', '贝叶斯准确率', 'SVM准确率']
    # data['准确率_score'] = [accuracy1, accuracy2, accuracy3]
    # data['精确率'] = ['KNN精确率', '贝叶斯精确率', 'SVM精确率']
    # data['精确率_score'] = [precision1, precision2, precision3]
    # data['召回率'] = ['KNN召回率', '贝叶斯召回率', 'SVM召回率']
    # data['召回率_score'] = [recall1, recall2, recall3]
    # data['F1值'] = ['KNNF1值', '贝叶斯F1值', 'SVMF1值']
    # data['F1值_score'] = [f1_1, f1_2, f1_3]
    #
    # data.to_csv('score.csv', encoding='utf-8-sig')
    
    # # 保存模型
    # joblib.dump(knn, 'knn_model.pkl')
    # joblib.dump(vectorizer, 'vectorizer.joblib')
    # # 保存LabelEncoder模型到文件中
    # with open("label_encoder.pkl", "wb") as f:
    #     pickle.dump(le, f)


def main2():
    def emjio_tihuan(x):
        x1 = str(x)
        x2 = re.sub('(\[.*?\])', "", x1)
        x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
        x4 = re.sub(r'\n', '', x3)
        return x4

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    # 定义机械压缩函数
    def yasuo(st):
        for i in range(1, int(len(st) / 2) + 1):
            for j in range(len(st)):
                if st[j:j + i] == st[j + i:j + 2 * i]:
                    k = j + i
                    while st[k:k + i] == st[k + i:k + 2 * i] and k < len(st):
                        k = k + i
                    st = st[:j] + st[k:]
        return st

    def get_cut_words(content_series):
        # 读入停用词表
        # 分词
        word_num = jieba.lcut(content_series, cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

        return ' '.join(word_num_selected)

    data = pd.read_excel('测试文本.xlsx')
    data['content'] = data['content'].apply(emjio_tihuan)
    data = data.dropna(subset=['content'], axis=0)
    # data['content'] = data['content'].apply(yasuo)
    data['分词'] = data['content'].apply(get_cut_words)

    knn = joblib.load('knn_model.pkl')
    vectorizer = joblib.load('vectorizer.joblib')

    with open("label_encoder.pkl", "rb") as f:
        label_encoder_loaded = pickle.load(f)

    X_vectorized = vectorizer.transform(data['分词'])

    # 使用模型进行分类
    predicted_label = knn.predict(X_vectorized)
    # 输出分类结果
    print('Predicted label:', predicted_label)

    #将数字解码回原始文本
    original_text = label_encoder_loaded.inverse_transform(predicted_label)

    data['search-area'] = original_text
    data.to_excel('预测文本.xlsx',index=False)


if __name__ == '__main__':
    main1()
    main2()
