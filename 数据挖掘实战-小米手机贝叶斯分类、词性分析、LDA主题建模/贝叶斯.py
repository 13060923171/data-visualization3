# 中文文本分类
import os
import jieba
import warnings
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
import re
import joblib


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def cut_words(text):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    seg_list = jieba.cut(text, cut_all=False)
    cut_words = (" ".join(seg_list))
    all_words = cut_words.split()
    for x in all_words:
        if len(x) >= 2 and x != '\r\n' and x != '\n' and is_all_chinese(x) == True and x not in stop_words:
            text_with_spaces += x + ' '
    text_with_spaces = text_with_spaces.strip(' ')
    return text_with_spaces


def loadfile(file_dir, label):
    """
    将路径下的所有文件加载
    :param file_dir: 保存txt文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    """
    words_list = []
    labels_list = []
    for file in file_dir:
        words_list.append(cut_words(file))
        labels_list.append(label)
    return words_list, labels_list


def emjio_tihuan(x):
    x1 = str(x)
    x2 = re.sub('(\[.*?\])', "", x1)
    x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
    x4 = re.sub(r'\n', '', x3)
    return x4

stop_words = []
with open('stopwords_cn.txt','r',encoding='utf-8')as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip().replace("'",""))

def emotion_type(x):
    x1 = str(x)
    if x1 == '好评':
        return "非负评论"
    elif x1 == '中评':
        return "非负评论"
    else:
        return "负面评论"


def train_data():
    df = pd.read_excel('评论信息.xlsx').iloc[:20000]
    df = df.drop_duplicates()
    df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
    df = df.dropna(subset=['评论内容'], axis=0)
    df['评论类型'] = df['评论类型'].apply(emotion_type)

    new_df = df['评论类型'].value_counts()
    x_data = list(new_df.index)

    train_words_list = []
    train_labels = []

    for x in x_data:
        data = df['评论内容'][df['评论类型'] == x]
        train_words_list1, train_labels1 = loadfile(data, x)
        train_words_list += train_words_list1
        train_labels += train_labels1
    return train_words_list,train_labels


def test_data():
    df = pd.read_excel('评论信息.xlsx').iloc[20000:]
    df = df.drop_duplicates()
    df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
    df = df.dropna(subset=['评论内容'], axis=0)
    df['评论类型'] = df['评论类型'].apply(emotion_type)

    new_df = df['评论类型'].value_counts()
    x_data = list(new_df.index)

    train_words_list = []
    train_labels = []

    for x in x_data:
        data = df['评论内容'][df['评论类型'] == x]
        train_words_list1, train_labels1 = loadfile(data, x)
        train_words_list += train_words_list1
        train_labels += train_labels1
    return train_words_list, train_labels


train_words_list,train_labels = train_data()
test_words_list,test_labels = test_data()


def main1():
    # 计算单词权重
    tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
    print(train_words_list)
    train_features = tf.fit_transform(train_words_list)
    # 上面fit过了，这里transform
    test_features = tf.transform(test_words_list)

    # 多项式贝叶斯分类器
    clf = MultinomialNB().fit(train_features, train_labels)
    predicted_labels=clf.predict(test_features)
    # 计算准确率
    print('情感准确率为：', metrics.accuracy_score(test_labels, predicted_labels))
    # 保存模型
    joblib.dump(clf, './model/nb_model.pkl')
    joblib.dump(tf, './model/vectorizer.joblib')


def main2(x):
    clf = joblib.load('./model/nb_model.pkl')
    tf = joblib.load('./model/vectorizer.joblib')
    train_words_list = []
    for i in x:
        train_words_list.append(cut_words(i))

    train_features = tf.transform(train_words_list)
    predicted_labels = clf.predict(train_features)
    return predicted_labels


if __name__ == '__main__':
    # main1()
    df = pd.read_excel('评论信息.xlsx')
    x_data = list(df['评论内容'])
    result = main2(x_data)
    df['分类结果'] = result
    df.to_csv('new_data.csv',encoding='utf-8-sig',index=False)