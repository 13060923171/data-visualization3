import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import multiprocessing
import os

def novel_demo(test):
    # 加载模型
    clf, vectorizer = joblib.load("../novel_svm_model.pkl")

    # 对剩余数据进行分类，并获取分类概率
    # 这里应为你的剩余数据
    remaining_data = test['分词']
    remaining_data_tfidf = vectorizer.transform(remaining_data)
    #分类概率 强度为1的概率
    predict_probas = clf.predict_proba(remaining_data_tfidf)[:, 1]
    predict = clf.predict(remaining_data_tfidf)

    data = pd.DataFrame()
    data['文本'] = test['文本']
    data['新颖性'] = predict
    data.to_excel('novel_data.xlsx',index=False)


def science_demo(test):
    # 加载模型
    clf, vectorizer = joblib.load("../science_svm_model.pkl")

    # 对剩余数据进行分类，并获取分类概率
    # 这里应为你的剩余数据
    remaining_data = test['分词']
    remaining_data_tfidf = vectorizer.transform(remaining_data)
    # 分类概率 强度为1的概率
    predict_probas = clf.predict_proba(remaining_data_tfidf)[:, 1]
    predict = clf.predict(remaining_data_tfidf)

    data = pd.DataFrame()
    data['文本'] = test['文本']
    data['科学价值'] = predict
    data.to_excel('science_data.xlsx', index=False)


def feasible_demo(test):
    # 加载模型
    clf, vectorizer = joblib.load("../feasible_svm_model.pkl")

    # 对剩余数据进行分类，并获取分类概率
    # 这里应为你的剩余数据
    remaining_data = test['分词']
    remaining_data_tfidf = vectorizer.transform(remaining_data)
    # 分类概率 强度为1的概率
    predict_probas = clf.predict_proba(remaining_data_tfidf)[:, 1]
    predict = clf.predict(remaining_data_tfidf)

    data = pd.DataFrame()
    data['文本'] = test['文本']
    data['可行性'] = predict
    data.to_excel('feasible_data.xlsx', index=False)

def research_demo(test):
    # 加载模型
    clf, vectorizer = joblib.load("../research_svm_model.pkl")

    # 对剩余数据进行分类，并获取分类概率
    # 这里应为你的剩余数据
    remaining_data = test['分词']
    remaining_data_tfidf = vectorizer.transform(remaining_data)
    # 分类概率 强度为1的概率
    predict_probas = clf.predict_proba(remaining_data_tfidf)[:, 1]
    predict = clf.predict(remaining_data_tfidf)

    data = pd.DataFrame()
    data['文本'] = test['文本']
    data['研究基础'] = predict
    data.to_excel('research_data.xlsx', index=False)


def quality_demo(test):
    # 加载模型
    clf, vectorizer = joblib.load("../quality_svm_model.pkl")

    # 对剩余数据进行分类，并获取分类概率
    # 这里应为你的剩余数据
    remaining_data = test['分词']
    remaining_data_tfidf = vectorizer.transform(remaining_data)
    # 分类概率 强度为1的概率
    predict_probas = clf.predict_proba(remaining_data_tfidf)[:, 1]
    predict = clf.predict(remaining_data_tfidf)

    data = pd.DataFrame()
    data['文本'] = test['文本']
    data['申报书质量'] = predict
    data.to_excel('quality_data.xlsx', index=False)


def main1():
    df1 = pd.read_excel('novel_data.xlsx')
    df2 = pd.read_excel('science_data.xlsx')
    df3 = pd.read_excel('feasible_data.xlsx')
    df4 = pd.read_excel('research_data.xlsx')
    df5 = pd.read_excel('quality_data.xlsx')

    df6 = pd.merge(df1, df2,on='文本')
    df7 = pd.merge(df3, df4, on='文本')
    df8 = pd.merge(df6, df7, on='文本')
    data = pd.merge(df8, df5, on='文本')

    # 更改列的顺序
    data = data[['文本','新颖性','科学价值','可行性','研究基础','申报书质量']]

    def tidai(x):
        x1 = str(x)
        if x1 == '2':
            return -1
        else:
            return x1

    df['新颖性'] = df['新颖性'].apply(tidai)
    df['科学价值'] = df['科学价值'].apply(tidai)
    df['可行性'] = df['可行性'].apply(tidai)
    df['研究基础'] = df['研究基础'].apply(tidai)
    df['申报书质量'] = df['申报书质量'].apply(tidai)

    data.to_excel('new_data.xlsx', index=False)
    os.remove('novel_data.xlsx')
    os.remove('science_data.xlsx')
    os.remove('feasible_data.xlsx')
    os.remove('research_data.xlsx')
    os.remove('quality_data.xlsx')


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    # 创建多进程
    p1 = multiprocessing.Process(target=novel_demo, args=(df,))
    p2 = multiprocessing.Process(target=science_demo, args=(df,))
    p3 = multiprocessing.Process(target=feasible_demo, args=(df,))
    p4 = multiprocessing.Process(target=research_demo, args=(df,))
    p5 = multiprocessing.Process(target=quality_demo, args=(df,))
    # 启动多进程
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    # 等待多进程的结束
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()

    main1()