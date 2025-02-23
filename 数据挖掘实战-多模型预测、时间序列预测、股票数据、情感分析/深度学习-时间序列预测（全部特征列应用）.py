import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import talib as ta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import Huber
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

# 1. 数据加载与预处理（修复缺失值填充方法）
# =================================================================
def convert_volume(volume):
    """
    将带有K/M/B后缀的交易量字符串转换为浮点数
    示例：
    "100K" → 100_000
    "1.5M" → 1_500_000
    "2B" → 2_000_000_000
    """
    if pd.isna(volume) or isinstance(volume, (int, float)):
        return volume

    volume = str(volume).upper()
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}

    for suffix, multiplier in multipliers.items():
        if suffix in volume:
            num = volume.replace(suffix, '').strip()
            return float(num) * multiplier

    return float(volume)  # 无后缀的情况

df1 = pd.read_excel('情感数据.xlsx')
df2 = pd.read_excel('比特币数据.xlsx')
# 应用线性插值
df2 = df2.interpolate()
data = pd.merge(df1,df2,left_on=['Date'],right_on=['日期'])
data = data.interpolate()

# 应用转换
data['交易量'] = data['交易量'].apply(convert_volume)
close = data['收盘价'].values

# 成交量相关
data['OBV'] = ta.OBV(close,data['交易量'])
data['VWAP'] = (data['交易量'] * (data['日最高'] + data['日最低'] + close) / 3).cumsum() / data['交易量'].cumsum()

# 统计特征
data['HL_PCT'] = (data['日最高'] - data['日最低']) / data['日最低'] * 100

# 添加对数变换处理价格序列
data['收盘价_log'] = np.log1p(data['收盘价'])


# 计算情感指标
data["Sentiment_Ratio"] = data["LABEL_1"] / (data["LABEL_0"] + data["LABEL_1"])  # 正面情感占比
data["Sentiment_Net"] = data["LABEL_1"] - data["LABEL_0"]  # 净情感数量

# 外部经济指标交互
data['GOLD_RATIO'] = data['收盘价'] / data['黄金价格:美元']
data['NASDAQ_RATIO'] = data['收盘价'] / data['纳斯达克综合指数']
data['Standard_RATIO'] = data['收盘价'] / data['标准普尔500指数']
data['Dow_RATIO'] = data['收盘价'] / data['道琼斯工业平均指数']



# 特征选择（目标变量单独处理）
features = ['开盘价', '日最高', '日最低', '交易量', '涨跌幅','OBV', 'VWAP','HL_PCT','收盘价_log','Sentiment_Ratio','Sentiment_Net','GOLD_RATIO','NASDAQ_RATIO','Standard_RATIO','Dow_RATIO']
target = '收盘价'

# 数据归一化（特征和目标变量分开归一化）
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

scaled_features = scaler_features.fit_transform(data[features])
scaled_target = scaler_target.fit_transform(data[[target]])

time_steps = 1
def create_dataset(features, target, time_steps):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:(i + time_steps), :])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_features, scaled_target, time_steps)

# 检查数据形状
print("X shape:", X.shape)  # 应为 (n_samples, time_steps, n_features)
print("y shape:", y.shape)  # 应为 (n_samples,)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. 模型构建与训练（修复输入形状错误）
# =================================================================
def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'BiLSTM':
        model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape))
    elif model_type == 'GRU':
        model.add(GRU(64, return_sequences=False, input_shape=input_shape))
    elif model_type == 'BiLSTM-GRU':
        model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape))
        model.add(GRU(32, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 定义早停机制
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 模型列表（输入形状修正为 (time_steps, n_features)）
input_shape = (X_train.shape[1], X_train.shape[2])
models = {
    'BiLSTM': build_model('BiLSTM', input_shape),
    'GRU': build_model('GRU', input_shape),
    'BiLSTM-GRU': build_model('BiLSTM-GRU', input_shape)
}

# 4. 训练与评估（修复反归一化逻辑）
# =================================================================
results = []
history_dict = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stopping]
    )
    history_dict[name] = history.history

    # 预测
    y_pred = model.predict(X_test)

    # 反归一化（仅需反归一化目标变量）
    y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler_target.inverse_transform(y_pred).flatten()

    # 计算指标
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mape = np.mean(np.abs((y_test - y_pred) / y_test))
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse,'MAPE': mape})

# 5. 结果保存与可视化
# =================================================================
# 保存指标到Excel
results_df = pd.DataFrame(results)
results_df.to_excel('model_comparison_1.xlsx', index=False)

# 可视化预测结果
plt.figure(figsize=(15, 6))
plt.plot(y_test_actual, label='Actual Price', linewidth=2)
for name in models.keys():
    y_pred_actual = scaler_target.inverse_transform(models[name].predict(X_test)).flatten()
    plt.plot(y_pred_actual, linestyle='--', label=f'{name} Prediction')
plt.title('Bitcoin Price Prediction Comparison')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.savefig('prediction_comparison_1.png')
plt.show()

# 输出最优模型
best_model = results_df.loc[results_df['MAE'].idxmin()]
print(f"\n最优模型: {best_model['Model']} (MAE={best_model['MAE']:.2f}, RMSE={best_model['RMSE']:.2f})")