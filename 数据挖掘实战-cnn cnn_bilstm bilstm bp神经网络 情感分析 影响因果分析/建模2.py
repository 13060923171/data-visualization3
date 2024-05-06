import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_excel('特征词.xlsx')
df['综合评分'] = round(df['综合评分'],2)
# df['综合评分'] = df['综合评分'].astype('int')
data = df[['聚类一','聚类二']]
targe = df['综合评分'].tolist()

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(data, targe, test_size=0.15, random_state=42)

# 数据归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# models = {
#     'Support Vector Classification': SVC(),
#     'Random Forest Classifier': RandomForestClassifier(),
#     'MLP Classifier': MLPClassifier(),
# }
#
# # K-fold交叉验证
# kf = KFold(n_splits=10, shuffle=True, random_state=42)
#
# # 用于存储模型表现的字典
# model_scores = {}
#
# svc_param_grid = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf'],
#     'gamma': ['scale', 'auto']
# }
#
# rfc_param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10]
# }
#
# mlp_param_grid = {
#     'hidden_layer_sizes': [(3,), (5,), (8,), (10,)],
#     'activation': ['relu', 'tanh'],
#     'alpha': [0.0001, 0.001, 0.01, 0.1]
# }
#
# # 创建评分字典
# scoring = {
#     'accuracy': 'accuracy',
#     'precision': make_scorer(precision_score, average='macro'),
#     'recall': make_scorer(recall_score, average='macro'),
#     'F1': make_scorer(f1_score, average='macro')
# }
#
#
#
# # 创建经过网格搜索调整后的模型
# svr_grid_search = GridSearchCV(SVC(), svc_param_grid,cv=kf,scoring=scoring, refit='F1')
# rfr_grid_search = GridSearchCV(RandomForestClassifier(), rfc_param_grid,cv=kf, scoring=scoring, refit='F1')
# mlp_grid_search = GridSearchCV(MLPClassifier(), mlp_param_grid,cv=kf, scoring=scoring, refit='F1')
#
# # 优化模型参数
# optimized_models = {
#     'Support Vector Regression': svr_grid_search,
#     'Random Forest Regressor': rfr_grid_search,
#     'MLP Regressor': mlp_grid_search,
# }
#
# # 训练并覆盖之前的模型
# for name, model in optimized_models.items():
#     model.fit(X_train, y_train)
#
# # 输出优化后的模型表现
# for name, model in optimized_models.items():
#     print(f"{name} Grid Search Results: ")
#     print(f"Best Parameters: {model.best_params_}")
#     # 获得最佳模型的评分结果
#     best_index = model.best_index_
#     print(f"Best Accuracy: {model.cv_results_['mean_test_accuracy'][best_index]}")
#     print(f"Best Precision: {model.cv_results_['mean_test_precision'][best_index]}")
#     print(f"Best Recall: {model.cv_results_['mean_test_recall'][best_index]}")
#     print(f"Best F1 Score: {model.cv_results_['mean_test_F1'][best_index]}\n")
#
# # 选择并评估表现最好的优化模型
# best_optimized_model_name = max(optimized_models, key=lambda k: optimized_models[k].best_score_)
# best_optimized_model = optimized_models[best_optimized_model_name]
# best_optimized_model.fit(X_train, y_train)
#
# # 使用测试集进行预测，并计算性能
# optimized_predictions = best_optimized_model.predict(X_test)
# # 更新评估指标为分类指标
# accuracy = accuracy_score(y_test, optimized_predictions)
# precision, recall, f1, _ = precision_recall_fscore_support(y_test, optimized_predictions, average='macro')
#
# print(f"Final Optimized Model: {best_optimized_model_name}")
# print(f"Test Set Accuracy (Optimized): {accuracy}")
# print(f"Test Set Precision (Optimized): {precision}")
# print(f"Test Set Recall (Optimized): {recall}")
# print(f"Test Set F1 Score (Optimized): {f1}")



# 训练模型
mlp = MLPRegressor(hidden_layer_sizes=(3,),
                   activation='relu',
                   alpha=0.0001,
                   max_iter=1000)
mlp.fit(X_train, y_train)

# 提取权重矩阵
weights = mlp.coefs_

def calculate_Rik(weights_input_hidden, weights_hidden_output):
    # 首先计算Rik产品矩阵
    Rik_matrix = np.dot(weights_input_hidden, weights_hidden_output)
    return Rik_matrix

# 使用先前从模型获取的权重
weights_input_hidden = weights[0]  # 输入层到隐藏层权重
weights_hidden_output = weights[-1]  # 隐藏层到输出层权重
print(weights_input_hidden)
print(weights_hidden_output)
# 计算Rik
Rik_values = calculate_Rik(weights_input_hidden, weights_hidden_output)
print(Rik_values)
#输入层到隐藏层权重矩阵
data1 = pd.DataFrame(weights_input_hidden, columns=['N1', 'N2','N3'])
data1['输入单元'] = ['生产设计','个人体验']
cols = data1.columns.tolist()
cols = cols[-1:] + cols[:-1]  # 将最后一列移到第一列
data1 = data1[cols]
data1.to_excel('输入层到隐藏层权重矩阵.xlsx')

#隐藏层到输出层权重
data2 = pd.DataFrame()
data2['输出单元'] = ['评分']
data2['N1'] = weights_hidden_output[0]
data2['N2'] = weights_hidden_output[1]
data2['N3'] = weights_hidden_output[2]
data2.to_excel('隐藏层到输出层权重.xlsx')

#相对影响程度
data3 = pd.DataFrame()
data3['一级特征'] = ['生产设计','个人体验']
data3['影响程度'] = Rik_values
data3.to_excel('相对影响程度.xlsx')

