import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
import joblib
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm


# ==================== 数据预处理 ====================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """特征工程处理器"""

    def __init__(self, window_size=5):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 添加技术指标
        X['MA5'] = X['收盘'].rolling(self.window_size).mean()
        X['MA10'] = X['收盘'].rolling(10).mean()
        X['Volatility'] = X['收盘'].rolling(5).std()

        # RSI计算
        delta = X['收盘'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        X['RSI'] = 100 - (100 / (1 + rs))

        # 滞后特征
        for lag in [1, 2, 3]:
            X[f'Lag_{lag}'] = X['收盘'].shift(lag)

        return X.dropna()


def train(n):
    # ==================== 数据加载与处理 ====================
    df1 = pd.read_csv(f'{x}.csv')
    df = df1[df1['板块'] == n]
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').set_index('日期')

    # 特征工程
    engineer = FeatureEngineer()
    df = engineer.transform(df)

    # 选择特征
    features = ['开盘', '成交量', '成交额', '振幅', '换手率', '最低', '最高',
                '涨跌幅', '涨跌额', '新闻强度', 'MA5', 'MA10', 'Volatility', 'RSI']
    target = '收盘'

    # 划分数据集
    split = int(0.8 * len(df))
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    # 标准化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    y_train = train_df[target].values
    y_test = test_df[target].values


    # ==================== 时间序列生成 ====================
    class SequenceGenerator:
        """时间序列数据生成器"""

        def __init__(self, time_steps=10):
            self.time_steps = time_steps

        def create_sequences(self, data, targets):
            X, y = [], []
            for i in range(len(data) - self.time_steps):
                X.append(data[i:i + self.time_steps])
                y.append(targets[i + self.time_steps])
            return np.array(X), np.array(y)


    # 动态调整时间步长
    time_steps = min(7, int(len(X_train) * 0.2))
    seq_gen = SequenceGenerator(time_steps)

    # 生成序列数据
    X_train_seq, y_train_seq = seq_gen.create_sequences(X_train, y_train)
    X_test_seq, y_test_seq = seq_gen.create_sequences(X_test, y_test)

    # ==================== 模型定义 ====================
    def build_lstm_model():
        """构建优化后的LSTM模型"""
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True,
                               kernel_regularizer=l2(0.01)),
                          input_shape=(time_steps, len(features))),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss='mse',
                      metrics=['mae'])
        return model


    def build_gru_model():
        """构建优化后的GRU模型"""
        model = Sequential([
            GRU(64, return_sequences=True,
                kernel_regularizer=l2(0.01),
                input_shape=(time_steps, len(features))),
            Dropout(0.3),
            GRU(32),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss='mse',
                      metrics=['mae'])
        return model


    def plot_training_history(model, title):
        plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 负号显示
        plt.figure(figsize=(10, 5))
        plt.plot(model.history_['loss'], label='训练损失')
        plt.plot(model.history_['val_loss'], label='验证损失')
        plt.title(f'{title} 训练过程')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./{x}/{n}-{title} 训练过程.png')
        plt.show()


    # ==================== 模型配置 ====================
    models = {
        'RandomForest': Pipeline([
            ('regressor', RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                random_state=42
            ))
        ]),

        'SVR': Pipeline([
            ('regressor', SVR(
                C=5,
                kernel='rbf',
                gamma='scale'
            ))
        ]),

        'LSTM': KerasRegressor(
            model=build_lstm_model,
            epochs=200,
            batch_size=16,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        ),

        'GRU': KerasRegressor(
            model=build_gru_model,
            epochs=200,
            batch_size=16,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )
    }

    # ==================== 模型训练与评估 ====================
    results = {}
    for name, model in models.items():
        try:
            if name in ['LSTM', 'GRU']:
                # 深度学习模型
                model.fit(X_train_seq, y_train_seq)
                y_pred = model.predict(X_test_seq)

            else:
                # 传统机器学习模型
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # 对齐测试数据
            y_true = y_test_seq if name in ['LSTM', 'GRU'] else y_test[time_steps:]
            y_pred = y_pred[:len(y_true)]  # 确保维度一致

            # 计算指标
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }

        except Exception as e:
            print(f"{name} 训练失败: {str(e)}")
            continue

    # LSTM训练过程
    plot_training_history(results['LSTM']['model'], 'LSTM')

    # GRU训练过程
    plot_training_history(results['GRU']['model'], 'GRU')

    # ==================== 结果可视化 ====================
    plt.figure(figsize=(15, 8))
    true_values = y_test_seq if 'LSTM' in results else y_test[time_steps:]
    plt.plot(true_values, label='真实值', linewidth=2)

    for name, res in results.items():
        plt.plot(res['predictions'], '--', linewidth=1.5, label=f'{name}预测值')

    plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示
    plt.title(f'{n}-模型预测对比')
    plt.xlabel('时间步')
    plt.ylabel('收盘价')
    plt.legend()
    plt.grid()
    plt.savefig(f'./{x}/{n}-模型预测对比.png')
    plt.show()

    # ==================== 模型对比 ====================
    metrics_df = pd.DataFrame({
        '板块':n,
        'Model': results.keys(),
        'MSE': [round(v['mse'],6) for v in results.values()],
        'MAE': [round(v['mae'],6) for v in results.values()],
        'R2': [round(v['r2'],6) for v in results.values()]
    }).sort_values('MSE')

    return metrics_df


if __name__ == '__main__':
    list_xw = ['国际新闻','科技新闻','娱乐新闻']
    for x in tqdm(list_xw):
        if not os.path.exists(f"./{x}"):
            os.mkdir(f"./{x}")
        data = pd.read_csv(f'{x}.csv')
        new_df = data['板块'].value_counts()
        list_bk = [x for x in new_df.index]
        list_df = []
        for b in tqdm(list_bk):
            metrics_df = train(b)
            list_df.append(metrics_df)

        new_df = pd.concat(list_df,axis=0)
        new_df.to_excel(f'{x}-模型指标数据.xlsx',index=False)