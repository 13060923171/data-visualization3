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
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

def demo(train,test):
    # 数据预处理，划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(train['fenci'], train['分类'], test_size=0.2, random_state=42)

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
    joblib.dump((clf, vectorizer), "svm_model.pkl")

    # 进行预测
    y_pred = clf.predict(X_test_tfidf)



    # 计算ROC曲线的数据点
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # 计算PR曲线的数据点
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

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
    plt.savefig('SVM_ROC曲线.png')
    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")

    plt.savefig('SVM_PR曲线.png')
    #创建分类报告
    clf_report = classification_report(y_test,y_pred)

    # 将分类报告保存为 .txt 文件
    with open('svm_classification_report.txt', 'w') as f:
        f.write(clf_report)

    # 加载模型
    clf, vectorizer = joblib.load("svm_model.pkl")

    # 对剩余数据进行分类，并获取分类概率
    # 这里应为你的剩余数据
    remaining_data = test['fenci']
    remaining_data_tfidf = vectorizer.transform(remaining_data)
    predict = clf.predict(remaining_data_tfidf)

    test['情感分类'] = predict

    def demo2(x):
        x1 = int(x)
        if x1 == 1:
            return 'pos'
        else:
            return 'neg'

    test['情感分类'] = test['情感分类'].apply(demo2)

    new_df = test['情感分类'].value_counts()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(16,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('情感占比分布')
    plt.tight_layout()
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.savefig('情感占比分布.png')

    test.to_excel('svm_test.xlsx',index=False)


if __name__ == '__main__':
    train = pd.read_excel('../train.xlsx')
    def demo1(x):
        x1 = str(x)
        if x1 == 'pos':
            return 1
        else:
            return 0


    train['分类'] = train['情感分类'].apply(demo1)

    test = pd.read_excel('../test.xlsx')

    demo(train,test)