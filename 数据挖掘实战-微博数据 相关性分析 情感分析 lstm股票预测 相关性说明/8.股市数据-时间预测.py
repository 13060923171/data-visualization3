import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


def prepare_data(df, time_step=1):
    X, y = [], []
    for i in range(len(df) - time_step - 1):
        X.append(df[i:(i + time_step), :])
        y.append(df[i + time_step, -1])
    return np.array(X), np.array(y)


def lstm_stock_prediction(data):
    # 选择相关特征
    features = ['开盘价', '最高价', '最低价', '收盘价']
    data = data[features]

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # 准备训练集和测试集
    time_step = 5  # 设置时间步长
    X, y = prepare_data(data_scaled, time_step)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, y_train, batch_size=1, epochs=50)

    # 预测
    y_pred = model.predict(X_test)

    # 反归一化预测结果
    y_test = y_test.reshape(-1, 1)
    y_pred = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], 3)), y_pred), axis=1))[:, -1]
    y_test = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 3)), y_test), axis=1))[:, -1]

    # 计算模型评分和均方误差
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f'R²: {r2}')
    print(f'MSE: {mse}')

    # 结果展示
    results = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
    results.to_excel('预测相关数据.xlsx')
    # 绘制百分比趋势柱状图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 绘制折线图
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, color='blue', label='实际值')
    plt.plot(y_pred, color='red', label='预测值')
    plt.title('实际值 vs 预测值')
    plt.xlabel('时间点')
    plt.ylabel('收盘价')
    plt.legend()
    plt.savefig('实际值 vs 预测值.png')
    plt.show()


# 示例数据
data = pd.read_excel('./data/股价表.xlsx')
lstm_stock_prediction(data)