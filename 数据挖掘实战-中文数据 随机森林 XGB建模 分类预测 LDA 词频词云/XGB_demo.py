import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
from sklearn.preprocessing import LabelEncoder


def polarity_demo(X_train, X_test, y_train, y_test):
    # 使用tf-idf进行特征抽取
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train['new_content'])
    X_test_tfidf = vectorizer.transform(X_test['new_content'])


    # 使用xgb进行训练
    params_grid = {'learning_rate': [0.1, 0.01],
                   'n_estimators': [100, 200],
                   'max_depth':  [3, 5]}
    xgb_model = xgb.XGBClassifier()
    clf = GridSearchCV(xgb_model, params_grid, cv=5)
    clf.fit(X_train_tfidf, y_train)

    # 进行预测
    y_pred = clf.predict(X_test_tfidf)
    #创建分类报告
    clf_report = classification_report(y_test,y_pred)

    # 将分类报告保存为 .txt 文件
    with open('./xgb/polarity_classification_report.txt', 'w') as f:
        f.write(clf_report)

    data = pd.DataFrame()
    data['content'] = X_test['content']
    data['location'] = X_test['location']
    data['情感极性'] = y_test
    data['预测-情感极性'] = y_pred
    data.to_csv('./xgb/polarity_xgb_data.csv',encoding='utf-8-sig',index=False)


def strength_demo(X_train, X_test, y_train, y_test):
    # 使用tf-idf进行特征抽取
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train['new_content'])
    X_test_tfidf = vectorizer.transform(X_test['new_content'])

    # 使用xgb进行训练
    params_grid = {'learning_rate': [0.1, 0.01],
                   'n_estimators': [100, 200],
                   'max_depth':  [3, 5]}
    xgb_model = xgb.XGBClassifier()
    clf = GridSearchCV(xgb_model, params_grid, cv=5)
    clf.fit(X_train_tfidf, y_train)

    # 进行预测
    y_pred = clf.predict(X_test_tfidf)
    # 创建分类报告
    clf_report = classification_report(y_test, y_pred)
    # 将分类报告保存为 .txt 文件
    with open('./xgb/strength_classification_report.txt', 'w') as f:
        f.write(clf_report)

    data = pd.DataFrame()
    data['content'] = X_test['content']
    data['location'] = X_test['location']
    data['情感强度'] = y_test
    data['预测-情感强度'] = y_pred
    data.to_csv('./xgb/strength_xgb_data.csv', encoding='utf-8-sig', index=False)


def motivation_demo(X_train, X_test, y_train, y_test):
    # 使用tf-idf进行特征抽取
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train['new_content'])
    X_test_tfidf = vectorizer.transform(X_test['new_content'])

    # 使用xgb进行训练
    params_grid = {'learning_rate': [0.1, 0.01],
                   'n_estimators': [100, 200],
                   'max_depth': [3, 5]}
    xgb_model = xgb.XGBClassifier()
    clf = GridSearchCV(xgb_model, params_grid, cv=5)
    clf.fit(X_train_tfidf, y_train)

    # 进行预测
    y_pred = clf.predict(X_test_tfidf)
    # 创建分类报告
    clf_report = classification_report(y_test, y_pred)
    # 将分类报告保存为 .txt 文件
    with open('./xgb/motivation_classification_report.txt', 'w') as f:
        f.write(clf_report)

    data = pd.DataFrame()
    data['content'] = X_test['content']
    data['location'] = X_test['location']
    data['情感分类'] = y_test
    data['预测-情感分类'] = y_pred
    data.to_csv('./xgb/motivation_xgb_data.csv', encoding='utf-8-sig', index=False)


def main1():
    data1 = pd.read_csv('./xgb/strength_xgb_data.csv')
    data2 = pd.read_csv('./xgb/polarity_xgb_data.csv')
    data3 = pd.read_csv('./xgb/motivation_xgb_data.csv')
    data4 = pd.merge(data1,data2,on='content')
    data5 = pd.merge(data4,data3,on='content')
    data5 = data5.drop(['location_x','location_y'],axis=1)

    def data_process(x):
        x1 = int(x)
        x2 = round((x1 / 10),2)
        return float(x2)
    def data_process1(x):
        if x == 0:
            return "消极"
        if x == 1:
            return "较为消极"
        if x == 2:
            return "中性"
        if x == 3:
            return "较为积极"
        if x == 4:
            return "积极"

    def data_process2(x):
        x1 = str(x)
        if x1 == "0":
            return "Anger"
        if x1 == "1":
            return "Fear"
        if x1 == "2":
            return "Sadness"
        if x1 == "3":
            return "Neutral"
        if x1 == "4":
            return "Joy"
        if x1 == "5":
            return "Love"

    data5['情感强度'] = data5['情感强度'].apply(data_process)
    data5['情感极性'] = data5['情感极性'].apply(data_process1)
    data5['情感分类'] = data5['情感分类'].apply(data_process2)
    data5['预测-情感强度'] = data5['预测-情感强度'].apply(data_process)
    data5['预测-情感极性'] = data5['预测-情感极性'].apply(data_process1)
    data5['预测-情感分类'] = data5['预测-情感分类'].apply(data_process2)
    data5.to_excel('./xgb/new_xgb_data.xlsx', index=False)


if __name__ == '__main__':
    df = pd.read_csv('new_data.csv')

    def data_process1(x):
        x1 = str(x)
        if x1 == "消极":
            return 0
        if x1 == "较为消极":
            return 1
        if x1 == "中立":
            return 2
        if x1 == "较为积极":
            return 3
        if x1 == "积极":
            return 4

    def data_process2(x):
        x1 = str(x)
        if x1 == "Anger":
            return 0
        if x1 == "Fear":
            return 1
        if x1 == "Sadness":
            return 2
        if x1 == "Neutral":
            return 3
        if x1 == "Joy":
            return 4
        if x1 == "Love":
            return 5
    df = df.dropna(subset=['new_content'],axis=0)
    df['情感强度'] = df['情感强度'] * 10
    df['情感强度'] = df['情感强度'].astype('int')
    df['情感极性'] = df['情感极性'].apply(data_process1)
    df['情感极性'] = df['情感极性'].astype('int')
    df['情感分类'] = df['情感分类'].apply(data_process2)
    df['情感分类'] = df['情感分类'].astype('int')

    # 将数据集划分为训练集、验证集、测试集
    # random_state=42 表示设定随机种子，以确保结果可复现
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df, df['情感强度'], test_size=0.25, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(df, df['情感极性'], test_size=0.25, random_state=42)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(df, df['情感分类'], test_size=0.25, random_state=42)

    # 创建多进程
    p1 = multiprocessing.Process(target=polarity_demo,args=(X_train2, X_test2, y_train2, y_test2))
    p2 = multiprocessing.Process(target=strength_demo,args=(X_train1, X_test1, y_train1, y_test1))
    p3 = multiprocessing.Process(target=motivation_demo,args=(X_train3, X_test3, y_train3, y_test3))
    # 启动多进程
    p1.start()
    p2.start()
    p3.start()
    # 等待多进程的结束
    p1.join()
    p2.join()
    p3.join()

    main1()