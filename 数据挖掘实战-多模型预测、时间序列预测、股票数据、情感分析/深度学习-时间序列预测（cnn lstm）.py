import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')
sns.set_style(style="whitegrid")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from pmdarima import auto_arima


# 1. 数据加载与预处理
def convert_volume(volume):
    if pd.isna(volume) or isinstance(volume, (int, float)):
        return volume

    volume = str(volume).upper()
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}

    for suffix, multiplier in multipliers.items():
        if suffix in volume:
            num = volume.replace(suffix, '').strip()
            return float(num) * multiplier
    return float(volume)


data = pd.read_excel('比特币数据.xlsx')
data = data.interpolate()
data['交易量'] = data['交易量'].apply(convert_volume)

features = ['开盘价', '日最高', '日最低', '交易量', '涨跌幅', '纳斯达克综合指数', '标准普尔500指数',
            '道琼斯工业平均指数', '上证综合指数', '黄金价格:美元']
target = '收盘价'

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

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# 2. 模型构建
def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    elif model_type == 'CNN':
        # 修正后的CNN结构
        model.add(Conv1D(64, kernel_size=1, activation='relu',
                        padding='same', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=1))  # 可选调整池化参数
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


input_shape = (X_train.shape[1], X_train.shape[2])
models = {
    'LSTM': build_model('LSTM', input_shape),
    'CNN': build_model('CNN', input_shape)
}

# 3. 训练与评估
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
results = []
history_dict = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stopping]
    )
    history_dict[name] = history.history

    y_pred = model.predict(X_test)
    y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler_target.inverse_transform(y_pred).flatten()

    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual))
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})


# 5. 结果保存与可视化
results_df = pd.DataFrame(results)
results_df.to_excel('model_comparison_2.xlsx', index=False)

plt.figure(figsize=(15, 6))
plt.plot(y_test_actual, label='Actual Price', linewidth=2)
for name in models.keys():
    y_pred_actual = scaler_target.inverse_transform(models[name].predict(X_test)).flatten()
    plt.plot(y_pred_actual, linestyle='--', label=f'{name} Prediction')
plt.title('Bitcoin Price Prediction Comparison')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.savefig('prediction_comparison_2.png')
plt.close()

best_model = results_df.loc[results_df['MAE'].idxmin()]
print(
    f"\n最优模型: {best_model['Model']} (MAE={best_model['MAE']:.2f}, RMSE={best_model['RMSE']:.2f}, MAPE={best_model['MAPE']:.2f}%)")