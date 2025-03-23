import pandas as pd
import numpy as np
import jieba
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense,Dense, Dropout,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")



# 数据准备保持不变
data = pd.read_csv('./lda/lda_data.csv')
X_train, X_test, y_train, y_test = train_test_split(data['fenci'], data['主题类型'], test_size=0.2, random_state=42)


# 词嵌入参数
max_features = 10000
max_len = 100

# 词典构建
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['fenci'].values)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# 序列填充
X_train_seq = pad_sequences(X_train_seq, maxlen=max_len)
X_test_seq = pad_sequences(X_test_seq, maxlen=max_len)


# 评估函数
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # 创建指标数据
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [accuracy, precision, recall, f1]

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#ccd5ae', '#e9edc9', '#fefae0', '#faedcd'])
    plt.ylim(0, 1.1)
    plt.title(f'Model Performance Metrics: {model_name}')
    plt.ylabel('Score')

    # 在柱子上方显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.savefig(f'cnn_model_comparison.png.png')
    plt.close()  # 关闭图形防止内存泄漏

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1

def cnn():
    # 构建 CNN 模型
    def build_cnn_model():
        model = Sequential()
        # 词嵌入层
        model.add(Embedding(max_features, 256, input_length=max_len))

        # 卷积层组
        model.add(Conv1D(128, 5, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        # 第二卷积层
        model.add(Conv1D(64, 3, activation='relu', padding='same'))
        model.add(GlobalMaxPooling1D())

        # 全连接层
        model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
        model.add(Dropout(0.3))

        # 输出层
        num_classes = len(np.unique(y_train))
        model.add(Dense(num_classes, activation='softmax'))

        # 优化器设置
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    cnn_model = build_cnn_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # 改为监控准确率
        patience=5,
        restore_best_weights=True
    )

    # 训练模型并保存训练历史
    history = cnn_model.fit(
        X_train_seq,
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练损失', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_loss'], label='验证损失', color='#ff7f0e', linestyle='--', linewidth=2)

    plt.title('cnn 训练损失曲线', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 标记最佳epoch（EarlyStopping恢复的权重）
    best_epoch = early_stopping.stopped_epoch - early_stopping.patience + 1
    if best_epoch >= 0:
        plt.axvline(best_epoch, color='red', linestyle=':', linewidth=1,
                    label=f'最佳epoch ({best_epoch + 1})')
        plt.legend()

    plt.tight_layout()
    plt.savefig('cnn训练损失曲线.png')
    plt.close()

    y_pred_probs = cnn_model.predict(X_test_seq)  # 输出概率矩阵
    y_pred_cnn = np.argmax(y_pred_probs, axis=1)
    metrics_cnn = evaluate_model(y_test, y_pred_cnn, 'cnn')


if __name__ == '__main__':
    cnn()

