import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
import jieba
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import svm
from tqdm import tqdm
import joblib
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer

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

    # score_list = []
    # for i in tqdm(range(1, 1000)):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(X_train, y_train)
    #     score = knn.score(X_test, y_test)
    #     score_list.append(score)
    #
    # best_k = score_list.index(max(score_list)) + 1
    # #406
    # print('Best K:', best_k)

    # 使用最优的K值训练模型并测试
    knn = KNeighborsClassifier(n_neighbors=406)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # 保存模型
    joblib.dump(knn, 'knn_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    # 保存LabelEncoder模型到文件中
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)


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
    # main1()
    main2()
