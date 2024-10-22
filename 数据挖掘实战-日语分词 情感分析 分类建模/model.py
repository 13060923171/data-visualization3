import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


def data_idea1():
    data1 = pd.read_excel('情感分类数据.xlsx')
    X = data1[['消极', '中立', '积极','发帖数量']]
    y = data1['结果']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        'decision_tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [None, 10, 20, 30,50,100],
                'min_samples_split': [2, 5, 10,15,20,50,100]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 102, 5, 10,15,20,50,100]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'svm': {
            'model': SVC(),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
    }


    best_models = {}

    for name, mp in param_grid.items():
        gs = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, n_jobs=-1)
        gs.fit(X_train, y_train)
        best_models[name] = gs.best_estimator_

        print(f"{name} best params: {gs.best_params_}")

    # 选择最佳的模型进行预测和评估
    with open('./demo1/model_1.txt', 'w', encoding='utf-8-sig') as f:
        for name, model in best_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            f.write(f"Model: {name}\n")
            f.write(f"准确率: {accuracy:.2f}\n")
            f.write(f"精确率: {precision:.2f}\n")
            f.write(f"召回率: {recall:.2f}\n")
            f.write(f"F1值: {f1:.2f}\n")
            f.write("=" * 30)

            # 计算每个点的FPR和TPR，这里假设正类是第二类（索引为1）
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
            plt.title(f'{name} Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(f'./demo1/{name}_ROC曲线.png')
            # 绘制PR曲线
            plt.figure()
            plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall')
            plt.legend(loc="lower right")
            plt.savefig(f'./demo1/{name}_PR曲线.png')


def data_idea2():
    data1 = pd.read_excel('情感分类数据.xlsx')
    # 新的特征数据框
    features = []
    # 索引遍历至倒数第二组的起始位置，因为需要三天的数据
    for i in range(len(data1) - 3 + 1):
        # 计算三天的情绪总和
        neg_sum = data1['消极'].iloc[i:i + 3].sum()
        neu_sum = data1['中立'].iloc[i:i + 3].sum()
        pos_sum = data1['积极'].iloc[i:i + 3].sum()
        total = data1['发帖数量'].iloc[i:i + 3].sum()

        # 第二天的结果作为目标值
        result = data1['结果'].iloc[i + 1]

        # 添加到特征列表
        features.append([neg_sum, neu_sum, pos_sum, total,result])
    # 转为DataFrame
    features_df = pd.DataFrame(features, columns=['neg_sum', 'neu_sum', 'pos_sum','total', 'result'])

    X = features_df[['neg_sum', 'neu_sum', 'pos_sum','total']]
    y = features_df['result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        'decision_tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [None, 10, 20, 30, 50, 100],
                'min_samples_split': [2, 5, 10, 15, 20, 50, 100]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 102, 5, 10, 15, 20, 50, 100]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'svm': {
            'model': SVC(),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
    }

    best_models = {}

    for name, mp in param_grid.items():
        gs = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, n_jobs=-1)
        gs.fit(X_train, y_train)
        best_models[name] = gs.best_estimator_

        print(f"{name} best params: {gs.best_params_}")

    # 选择最佳的模型进行预测和评估
    with open('./demo2/model_1.txt', 'w', encoding='utf-8-sig') as f:
        for name, model in best_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            f.write(f"Model: {name}\n")
            f.write(f"准确率: {accuracy:.2f}\n")
            f.write(f"精确率: {precision:.2f}\n")
            f.write(f"召回率: {recall:.2f}\n")
            f.write(f"F1值: {f1:.2f}\n")
            f.write("=" * 30)

            # 计算每个点的FPR和TPR，这里假设正类是第二类（索引为1）
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
            plt.title(f'{name} Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(f'./demo2/{name}_ROC曲线.png')
            # 绘制PR曲线
            plt.figure()
            plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall')
            plt.legend(loc="lower right")
            plt.savefig(f'./demo2/{name}_PR曲线.png')


if __name__ == '__main__':
    data_idea1()
    data_idea2()