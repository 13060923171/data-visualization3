import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from config import modify_weights
from tqdm import tqdm
df = pd.read_excel('特征词.xlsx')
df['综合评分'] = round(df['综合评分'],1)
data = df[['生产设计','售前服务','个人体验']]
# df['一级特征'] = df['生产设计'] + df['售前服务'] + df['个人体验']
# data = df[['一级特征']]
targe = df['综合评分'].tolist()

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(data, targe, test_size=0.3, random_state=42)

# 数据归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 选择模型
models = {
    'Support Vector Regression': SVR(),
    'Random Forest Regressor': RandomForestRegressor(),
    'MLP Regressor': MLPRegressor(hidden_layer_sizes=()),
}

# K-fold交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 用于存储模型表现的字典
model_scores = {}

# 创建每个模型的参数网格
svr_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

rfr_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

mlp_param_grid = {
    'hidden_layer_sizes': [(3,),(5,),(8,),(10,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01, 0.1]
}

# 创建评分字典
scoring = {
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
    'R2': 'r2'
}



# 创建经过网格搜索调整后的模型
svr_grid_search = GridSearchCV(SVR(), svr_param_grid, cv=kf,scoring=scoring, refit='R2')
rfr_grid_search = GridSearchCV(RandomForestRegressor(), rfr_param_grid, cv=kf, scoring=scoring, refit='R2')
mlp_grid_search = GridSearchCV(MLPRegressor(), mlp_param_grid, cv=kf, scoring=scoring, refit='R2')

# 优化模型参数
optimized_models = {
    'Support Vector Regression': svr_grid_search,
    'Random Forest Regressor': rfr_grid_search,
    'MLP Regressor': mlp_grid_search,
}

# 训练并覆盖之前的模型
for name, model in optimized_models.items():
    model.fit(X_train_scaled, y_train)

# 输出优化后的模型表现
for name, model in optimized_models.items():
    print(f"{name} Grid Search Results: ")
    print(f"Best Parameters: {model.best_params_}")
    # 获得最佳模型的评分结果
    best_index = model.best_index_
    print(f"Best Mean Squared Error: {-model.cv_results_['mean_test_MSE'][best_index]}")
    print(f"Best R² Score: {model.cv_results_['mean_test_R2'][best_index]}\n")

# 选择并评估表现最好的优化模型
best_optimized_model_name = max(optimized_models, key=lambda k: optimized_models[k].best_score_)
best_optimized_model = optimized_models[best_optimized_model_name]
best_optimized_model.fit(X_train_scaled, y_train)

# 使用测试集进行预测，并计算性能
optimized_predictions = best_optimized_model.predict(X_test_scaled)
optimized_final_mse = mean_squared_error(y_test, optimized_predictions)
optimized_final_r2 = r2_score(y_test, optimized_predictions)

print(f"Final Optimized Model: {best_optimized_model_name}")
print(f"Test Set Mean Squared Error (Optimized): {optimized_final_mse}")
print(f"Test Set R² Score (Optimized): {optimized_final_r2}")



max_iterations = 1000
weights_below_one = False

for iteration in tqdm(range(max_iterations), desc="训练进度"):
    mlp = MLPRegressor(hidden_layer_sizes=(3,),
                       activation='tanh',
                       alpha=0.1,
                       max_iter=1000)
    mlp.fit(X_train_scaled, y_train)

    # 提取权重矩阵
    weights = mlp.coefs_
    weights = modify_weights(weights)

    # 检查所有权重是否都小于1
    weights_below_one = all(abs(weight) < 1 for layer in weights for weight in np.nditer(layer))

    if weights_below_one:
        print(f"终止于迭代 {iteration + 1}: 所有权重值都小于1.")
        break

if not weights_below_one:
    print(f"已完成 {max_iterations} 次迭代.")

if weights_below_one:
    def calculate_Rik(weights_input_hidden, weights_hidden_output):
        # 首先计算Rik产品矩阵
        Rik_matrix = np.dot(weights_input_hidden, weights_hidden_output)
        return Rik_matrix

    # 使用先前从模型获取的权重
    weights_input_hidden = weights[0]  # 输入层到隐藏层权重
    weights_hidden_output = weights[-1]  # 隐藏层到输出层权重

    # # 计算Rik
    Rik_values = calculate_Rik(weights_input_hidden, weights_hidden_output)

    #输入层到隐藏层权重矩阵
    data1 = pd.DataFrame(weights_input_hidden, columns=['N1', 'N2','N3'])
    data1['输入单元'] = ['生产设计','售前服务','个人体验']
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
    data3['一级特征'] = ['生产设计','售前服务','个人体验']
    data3['影响程度'] = Rik_values
    data3.to_excel('相对影响程度.xlsx')

