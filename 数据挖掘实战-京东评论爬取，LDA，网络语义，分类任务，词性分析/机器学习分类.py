import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix,precision_recall_curve
from sklearn.pipeline import Pipeline
import jieba
from sklearn.metrics import accuracy_score
from sklearn import svm
from tqdm import tqdm
import joblib
import re
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import numpy as np

df = pd.read_csv('new_data.csv')
new_df = df.dropna(subset=['分词'], axis=0)
stop_words = []

with open("stopwords_cn.txt", 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


def main1():
    # 创建LabelEncoder对象
    le = LabelEncoder()
    # 使用LabelEncoder对象对分类变量进行编码
    y = le.fit_transform(new_df['class'])

    # 将分词后的中文文本转换为数字类型
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(new_df['分词'])

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

    # 训练和预测 - SVM
    clf_svm = SVC()
    clf_svm.fit(X_train, y_train)
    y_pred_svm = clf_svm.predict(X_test)
    print('SVM accuracy:', accuracy_score(y_test, y_pred_svm))

    ##逻辑回归分类
    clf_lr = LogisticRegression()
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)
    print('LogisticRegression accuracy:', accuracy_score(y_test, y_pred_lr))

    # 计算准确率
    accuracy1 = accuracy_score(y_test,y_pred_svm)
    accuracy2 = accuracy_score(y_test,y_pred_lr)

    # 计算精确率
    precision1 = precision_score(y_test,y_pred_svm, average='macro')
    precision2 = precision_score(y_test,y_pred_lr, average='macro')

    # 计算召回率
    recall1 = recall_score(y_test,y_pred_svm, average='macro')
    recall2 = recall_score(y_test,y_pred_lr, average='macro')

    # 计算F1值
    f1_1 = f1_score(y_test,y_pred_svm, average='macro')
    f1_2 = f1_score(y_test,y_pred_lr, average='macro')


    data = pd.DataFrame()
    data['准确率'] = ['SVM准确率', '逻辑回归准确率']
    data['准确率_score'] = [accuracy1, accuracy2]
    data['精确率'] = ['SVM精确率', '逻辑回归精确率']
    data['精确率_score'] = [precision1, precision2]
    data['召回率'] = ['SVM召回率', '逻辑回归召回率']
    data['召回率_score'] = [recall1, recall2]
    data['F1值'] = ['SVMF1值', '逻辑回归F1值']
    data['F1值_score'] = [f1_1, f1_2]

    data.to_csv('score.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main1()

