import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV, cross_val_predict
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. 数据加载与预处理
df = pd.read_csv('19-24.csv')
df.drop(columns=['序号', 'TEAM'], inplace=True)  # 移除非特征列
X = df.drop(columns=['result'])
y = df['result']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 特征筛选（仅使用CFS方法）
cfs_selector = SelectKBest(mutual_info_classif, k=10)  # 选择前10个重要特征
X_train_fs = cfs_selector.fit_transform(X_train, y_train)
selected_features = X.columns[cfs_selector.get_support()].tolist()
X_test_fs = X_test[selected_features]

print("Selected Features:", selected_features)

# 3. 模型训练与调参
models = {
    'RF': (RandomForestClassifier(), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
    'DT': (DecisionTreeClassifier(), {'max_depth': [5, 10, None]}),
    'KNN': (Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]),
            {'knn__n_neighbors': [3, 5, 7]}),
    'XGB': (XGBClassifier(), {'learning_rate': [0.1, 0.3], 'max_depth': [3, 5]}),
    'LGBM': (LGBMClassifier(), {'num_leaves': [31, 63], 'learning_rate': [0.1, 0.3]}),
    'Ada': (AdaBoostClassifier(), {'n_estimators': [50, 100], 'learning_rate': [0.8, 1.0]}),
    'LR': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]})
}

best_models = {}
for name, (model, params) in models.items():
    clf = RandomizedSearchCV(model, params, n_iter=2, cv=3, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train_fs, y_train)
    best_models[name] = clf.best_estimator_
    print(f"{name} Best Params: {clf.best_params_}")

# 4. Stacking模型
base_models = [
    ('rf', best_models['RF']),
    ('xgb', best_models['XGB']),
    ('lgbm', best_models['LGBM'])
]
meta_model = LogisticRegression()
stacking = StackingClassifier(estimators=base_models, final_estimator=meta_model)
stacking.fit(X_train_fs, y_train)

# 5. 模型评估
final_models = {'Stacking': stacking}
final_models.update(best_models)

plt.figure(figsize=(10, 8))
list_name = []
list_accuracy = []
list_precision = []
list_recall = []
list_f1 = []
for name, model in final_models.items():
    # 10折交叉验证
    y_pred = cross_val_predict(model, X_train_fs, y_train, cv=10, method='predict_proba')[:, 1]
    fpr, tpr, _ = roc_curve(y_train, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    # 训练测试集评估
    model.fit(X_train_fs, y_train)
    y_test_pred = model.predict(X_test_fs)
    print(f"\n{name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1: {f1_score(y_test, y_test_pred):.4f}")
    list_name.append(name)
    list_accuracy.append(accuracy_score(y_test, y_test_pred))
    list_precision.append(precision_score(y_test, y_test_pred))
    list_recall.append(recall_score(y_test, y_test_pred))
    list_f1.append(f1_score(y_test, y_test_pred))

data = pd.DataFrame()
data['name'] = list_name
data['accuracy'] = list_accuracy
data['precision'] = list_precision
data['recall'] = list_recall
data['f1'] = list_f1
data.to_excel(f"评估指标数据.xlsx",index=False)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.savefig(f'ROC Curve Comparison.png')
plt.show()