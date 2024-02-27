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


def novel_demo(train):
    # 数据预处理，划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(train['分词'], train['新颖性'], test_size=0.25, random_state=42)

    # 使用tf-idf进行特征抽取
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 使用SVM进行训练
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, params_grid, cv=5)
    clf.fit(X_train_tfidf, y_train)

    # 保存训练好的模型
    joblib.dump((clf, vectorizer), "novel_svm_model.pkl")

    # 进行预测
    y_pred = clf.predict(X_test_tfidf)
    #创建分类报告
    clf_report = classification_report(y_test,y_pred)

    # 将分类报告保存为 .txt 文件
    with open('novel_classification_report.txt', 'w') as f:
        f.write(clf_report)


def science_demo(train):
    # 数据预处理，划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(train['分词'], train['科学价值'], test_size=0.25, random_state=42)

    # 使用tf-idf进行特征抽取
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 使用SVM进行训练
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, params_grid, cv=5)
    clf.fit(X_train_tfidf, y_train)

    # 保存训练好的模型
    joblib.dump((clf, vectorizer), "science_svm_model.pkl")

    # 进行预测
    y_pred = clf.predict(X_test_tfidf)
    # 创建分类报告
    clf_report = classification_report(y_test, y_pred)

    # 将分类报告保存为 .txt 文件
    with open('science_classification_report.txt', 'w') as f:
        f.write(clf_report)


def feasible_demo(train):
    # 数据预处理，划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(train['分词'], train['可行性'], test_size=0.25, random_state=42)

    # 使用tf-idf进行特征抽取
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 使用SVM进行训练
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, params_grid, cv=5)
    clf.fit(X_train_tfidf, y_train)

    # 保存训练好的模型
    joblib.dump((clf, vectorizer), "feasible_svm_model.pkl")

    # 进行预测
    y_pred = clf.predict(X_test_tfidf)
    # 创建分类报告
    clf_report = classification_report(y_test, y_pred)

    # 将分类报告保存为 .txt 文件
    with open('feasible_classification_report.txt', 'w') as f:
        f.write(clf_report)


def research_demo(train):
    # 数据预处理，划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(train['分词'], train['研究基础'], test_size=0.25, random_state=42)

    # 使用tf-idf进行特征抽取
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 使用SVM进行训练
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, params_grid, cv=5)
    clf.fit(X_train_tfidf, y_train)

    # 保存训练好的模型
    joblib.dump((clf, vectorizer), "research_svm_model.pkl")

    # 进行预测
    y_pred = clf.predict(X_test_tfidf)
    # 创建分类报告
    clf_report = classification_report(y_test, y_pred)

    # 将分类报告保存为 .txt 文件
    with open('research_classification_report.txt', 'w') as f:
        f.write(clf_report)


def quality_demo(train):
    # 数据预处理，划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(train['分词'], train['申报书质量'], test_size=0.25, random_state=42)

    # 使用tf-idf进行特征抽取
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 使用SVM进行训练
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, params_grid, cv=5)
    clf.fit(X_train_tfidf, y_train)

    # 保存训练好的模型
    joblib.dump((clf, vectorizer), "quality_svm_model.pkl")

    # 进行预测
    y_pred = clf.predict(X_test_tfidf)
    # 创建分类报告
    clf_report = classification_report(y_test, y_pred)

    # 将分类报告保存为 .txt 文件
    with open('quality_classification_report.txt', 'w') as f:
        f.write(clf_report)


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    def tidai(x):
        x1 = str(x)
        if x1 == '-1':
            return 2
        else:
            return x1

    df['新颖性'] = df['新颖性'].apply(tidai)
    df['新颖性'] = df['新颖性'].astype('int')
    df['科学价值'] = df['科学价值'].apply(tidai)
    df['科学价值'] = df['科学价值'].astype('int')
    df['可行性'] = df['可行性'].apply(tidai)
    df['可行性'] = df['可行性'].astype('int')
    df['研究基础'] = df['研究基础'].apply(tidai)
    df['研究基础'] = df['研究基础'].astype('int')
    df['申报书质量'] = df['申报书质量'].apply(tidai)
    df['申报书质量'] = df['申报书质量'].astype('int')

    # 创建多进程
    p1 = multiprocessing.Process(target=novel_demo,args=(df,))
    p2 = multiprocessing.Process(target=science_demo,args=(df,))
    p3 = multiprocessing.Process(target=feasible_demo,args=(df,))
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
