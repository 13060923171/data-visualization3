import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tqdm import tqdm
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

df = pd.read_csv('data.csv')
df['price'] = df['price'] / 1000
df = df.drop(['rental_method','housetype'],axis=1)


def orient_process(x):
    x1 = str(x).split(" ")
    return x1[0].strip(" ")


df['orient'] = df['orient'].apply(orient_process)


def floor_process(x):
    if x == '低楼层':
        return 0
    if x == '中楼层':
        return 1
    if x == '高楼层':
        return 2


df['floor'] = df['floor'].apply(floor_process)

le = LabelEncoder()

df['orient'] = le.fit_transform(df['orient'])
df['district'] = le.fit_transform(df['district'])
df['city'] = le.fit_transform(df['city'])

# 计算四分位数
Q1 = df['price'].quantile(0.25)  # 第一四分位数
Q3 = df['price'].quantile(0.75)  # 第三四分位数
IQR = Q3 - Q1  # 四分位距

# 定义上下限，一般取1.5倍IQR作为异常的标准
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 去掉异常值
df_ = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
df1 = df_.drop(['price'],axis=1)


# X = df1
# y = df_['price']
#
# model = LogisticRegression()
# rfe = RFE(model, n_features_to_select=5)  # 选择5个最重要的特征
# rfe.fit(X, y)
#
# selected_features = X.columns[rfe.support_]
# # Index(['district', 'area', 'orient', 'floor', 'city'], dtype='object')
# print(selected_features)  # 输出被选中的特征

X2 = df1[['district', 'area', 'orient', 'floor', 'city']]
y1 = df_['price']

# 假设 X 是你的特征数据，y 是目标变量
X_train, X_test, y_train, y_test = train_test_split(X2, y1, test_size=0.2, random_state=42)

# 初始化 StandardScaler
scaler = StandardScaler()

# 对训练集进行拟合并标准化
X_train_scaled = scaler.fit_transform(X_train)

# 对测试集进行标准化（使用在训练集上拟合的 scaler）
X_test_scaled = scaler.transform(X_test)


ensemble_model = VotingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('gb', GradientBoostingRegressor(learning_rate=0.1, n_estimators=200)),
        ('xgb', XGBRegressor())

    ]
)

ensemble_model.fit(X_train_scaled, y_train)
y_pred = ensemble_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
results = []
results.append({
    'Model': 'Ensemble',
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2
})
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_excel('model.xlsx',index=False)

# # 各个模型的初始化和参数网格
# models = {
#     # 'Linear Regression': LinearRegression(),
#     # 'Ridge': Ridge(),
#     # 'Lasso': Lasso(),
#     # 'Random Forest': RandomForestRegressor(),
#     # 'Gradient Boosting': GradientBoostingRegressor(),
#     'MLP':MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
# }
#
# param_grids = {
#     # 'Linear Regression': {},
#     # 'Ridge': {'alpha': [0.1, 1.0, 10.0]},
#     # 'Lasso': {'alpha': [0.1, 1.0, 10.0]},
#     # 'Random Forest': {'n_estimators': [50, 100, 200]},
#     # 'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
#     'MLP':{'alpha': [0.0001, 0.001, 0.01],'hidden_layer_sizes': [(50,), (100,), (100, 50)]}
# }
#
#
# results = []
#
# for name, model in tqdm(models.items()):
#     grid_search = GridSearchCV(model, param_grids[name], scoring='neg_mean_squared_error', cv=5)
#     grid_search.fit(X_train_scaled, y_train)
#     best_model = grid_search.best_estimator_
#
#     # 预测和评估
#     y_pred = best_model.predict(X_test_scaled)
#
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
#
#     results.append({
#         'Model': name,
#         'Best Params': grid_search.best_params_,
#         'MSE': mse,
#         'RMSE': rmse,
#         'R2': r2
#     })
#
# # 将结果整理为 DataFrame
# results_df = pd.DataFrame(results)
# print(results_df)
# results_df.to_excel('model.xlsx',index=False)
#
#
# # 选择 R2 最高的模型作为最优模型
# best_model_info = results_df.loc[results_df['R2'].idxmax()]
# best_model_name = best_model_info['Model']
# best_params = best_model_info['Best Params']
#
# print(f"最优模型是: {best_model_name}，参数是: {best_params}")
#
#
# # 保存 LabelEncoder
# joblib.dump(le, 'label_encoder.pkl')
#
# # 保存 StandardScaler
# joblib.dump(scaler, 'scaler.pkl')
#
# # 保存最优模型
# best_model = models[best_model_name].set_params(**best_params)
# best_model.fit(X_train_scaled, y_train)  # 在整个训练集上重新拟合最优模型
#
# # 预测和评估
# y_pred = best_model.predict(X_test_scaled[:100])
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
# # 创建一个折线图来比较实际值 (y_test) 和预测值 (y_pred)
# plt.figure(figsize=(12, 6), dpi=500)
#
# # 绘制实际值
# plt.plot(y_test.reset_index(drop=True)[:100], label='Actual', color='b')
#
# # 绘制预测值
# plt.plot(y_pred, label='Predicted', color='r', linestyle='--')
#
# # 添加图例和标题
# plt.legend()
# plt.title(f"Actual vs Predicted values using {best_model_name}")
# plt.xlabel("Sample Index")
# plt.ylabel("Price")
# plt.savefig(f"Actual vs Predicted values using {best_model_name}.png")
# # 显示图像
# plt.show()
#
# joblib.dump(best_model, f'{best_model_name}_model.pkl')