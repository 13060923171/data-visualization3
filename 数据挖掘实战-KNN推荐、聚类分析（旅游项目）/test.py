# 导入所需的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization
import numpy as np
# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义knn的参数范围和评价函数
knn_params = {'n_neighbors': (1, 20)}
def knn_eval(n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=int(n_neighbors))
    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)

# 定义贝叶斯的评价函数
def bayes_eval(c):
    bayes = MultinomialNB(alpha=c)
    bayes.fit(X_train, y_train)
    return bayes.score(X_test, y_test)

# 定义svm的参数范围和评价函数
svm_params = {'C': (0.1, 10), 'gamma': (0.01, 1)}

def svm_eval(C, gamma):
    svm = SVC(C=C, gamma=gamma)
    svm.fit(X_train, y_train)
    return svm.score(X_test, y_test)



for i in np.arange(0.1, 0.5, 0.1):
    print(i)