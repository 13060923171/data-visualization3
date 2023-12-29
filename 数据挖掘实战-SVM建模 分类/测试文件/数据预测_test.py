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


def polarity_demo(test):
    # 加载模型
    clf, vectorizer = joblib.load("polarity_svm_model.pkl")

    # 对剩余数据进行分类，并获取分类概率
    # 这里应为你的剩余数据
    remaining_data = test['分词']
    remaining_data_tfidf = vectorizer.transform(remaining_data)
    #分类概率 强度为1的概率
    predict_probas = clf.predict_proba(remaining_data_tfidf)[:, 1]
    predict = clf.predict(remaining_data_tfidf)


    test['极性(-1/1)'] = predict
    test['polarity'] = predict_probas
    test = test.drop(['强度（1/0）','动机（1/0）','是否为训练集','分词'],axis=1)
    test.to_excel('polarity_data.xlsx',index=False)


def strength_demo(test):
    # 加载模型
    clf, vectorizer = joblib.load("strength_svm_model.pkl")

    # 对剩余数据进行分类，并获取分类概率
    # 这里应为你的剩余数据
    remaining_data = test['分词']
    remaining_data_tfidf = vectorizer.transform(remaining_data)
    # 分类概率 强度为1的概率
    predict_probas = clf.predict_proba(remaining_data_tfidf)[:, 1]
    predict = clf.predict(remaining_data_tfidf)

    test['强度（1/0）'] = predict
    test['strength'] = predict_probas
    test = test.drop(['极性(-1/1)', '动机（1/0）', '是否为训练集', '分词'], axis=1)
    test.to_excel('strength_data.xlsx', index=False)


def motivation_demo(test):
    # 加载模型
    clf, vectorizer = joblib.load("motivation_svm_model.pkl")

    # 对剩余数据进行分类，并获取分类概率
    # 这里应为你的剩余数据
    remaining_data = test['分词']
    remaining_data_tfidf = vectorizer.transform(remaining_data)
    # 分类概率 强度为1的概率
    predict_probas = clf.predict_proba(remaining_data_tfidf)[:, 1]
    predict = clf.predict(remaining_data_tfidf)

    test['动机（1/0）'] = predict
    test['motivation'] = predict_probas
    test = test.drop(['极性(-1/1)', '强度（1/0）', '是否为训练集', '分词'], axis=1)
    test.to_excel('motivation_data.xlsx', index=False)


def main1():
    df1 = pd.read_excel('strength_data.xlsx')
    df2 = pd.read_excel('polarity_data.xlsx')
    df3 = pd.read_excel('motivation_data.xlsx')
    df4 = pd.merge(df1,df2,on='评论内容')
    data = pd.merge(df4,df3,on='评论内容')
    data = data.drop(['店铺名称_x','店铺名称_y'],axis=1)
    # 更改列的顺序
    data = data[['店铺名称', '评论内容','强度（1/0）','极性(-1/1)','动机（1/0）','strength','polarity','motivation']]

    def strength_type(x):
        x1 = float(x)
        if x1 >= 0.5:
            return '有明显情绪表达'
        else:
            return '无情绪表达'

    def motivation_type(x):
        x1 = float(x)
        if x1 >= 0.5:
            return '有明显动机'
        else:
            return '无明显动机'

    def polarity_type(x):
        x1 = float(x)
        if x1 > 0.51:
            return '积极'
        elif 0.49 <= x1 <= 0.51:
            return '中性'
        else:
            return '消极'

    data['情绪表达'] = data['strength'].apply(strength_type)
    data['动机表达'] = data['motivation'].apply(motivation_type)
    data['情绪分类'] = data['polarity'].apply(polarity_type)

    new_df = data['情绪表达'].value_counts()
    new_df.to_excel('微博文本情绪强度分类统计.xlsx')

    new_df1 = data['动机表达'].value_counts()
    new_df1.to_excel('微博文本动机强度分类统计.xlsx')

    new_df2 = data['情绪分类'].value_counts()
    new_df2.to_excel('微博文本情绪分类统计.xlsx')

    data.to_excel('new_data.xlsx', index=False)


if __name__ == '__main__':
    df = pd.read_csv('data.csv')

    def demo(x):
        x1 = str(x)
        if x1 == "1.0" or x1 == "0.0":
            return 1
        else:
            return 0

    df['是否为训练集'] = df['强度（1/0）'].apply(demo)
    train = df[df['是否为训练集'] == 1]
    test = df[df['是否为训练集'] == 0]

    # 创建多进程
    p1 = multiprocessing.Process(target=polarity_demo,args=(test,))
    p2 = multiprocessing.Process(target=strength_demo,args=(test,))
    p3 = multiprocessing.Process(target=motivation_demo,args=(test,))
    # 启动多进程
    p1.start()
    p2.start()
    p3.start()
    # 等待多进程的结束
    p1.join()
    p2.join()
    p3.join()

    main1()