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


def demo(train,test):
    # 数据预处理，划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(train['fenci'], train['情感标签'], test_size=0.25, random_state=42)

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
    joblib.dump((clf, vectorizer), "./豆瓣/svm_model.pkl")

    # 进行预测
    y_pred = clf.predict(X_test_tfidf)
    #创建分类报告
    clf_report = classification_report(y_test,y_pred)

    # 将分类报告保存为 .txt 文件
    with open('./豆瓣/svm_classification_report.txt', 'w') as f:
        f.write(clf_report)

    # 加载模型
    clf, vectorizer = joblib.load("./豆瓣/svm_model.pkl")

    # 对剩余数据进行分类，并获取分类概率
    # 这里应为你的剩余数据
    remaining_data = test['fenci']
    remaining_data_tfidf = vectorizer.transform(remaining_data)
    predict = clf.predict(remaining_data_tfidf)

    test['情感标签'] = predict


    data = pd.concat([train,test],axis=0)
    data['情感标签'] = data['情感标签'].astype('int')
    def emotion_type(x):
        if x == 0:
            return '中立'
        elif x == 1:
            return '正面'
        else:
            return '负面'

    data['情感标签'] = data['情感标签'].apply(emotion_type)
    data.to_csv('./豆瓣/new_data.csv',encoding='utf-8-sig',index=False)




if __name__ == '__main__':
    train = pd.read_csv('./豆瓣/train_data.csv')
    test = pd.read_csv('./豆瓣/test_data.csv')

    demo(train,test)
