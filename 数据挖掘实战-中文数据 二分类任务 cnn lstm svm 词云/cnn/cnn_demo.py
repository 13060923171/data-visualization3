import pandas as pd
import numpy as np
import matplotlib
from joblib import dump, load
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

df = pd.read_excel("../train.xlsx")
df['fenci'] = df['fenci'].astype('str')


def demo(x):
    x1 = str(x)
    if x1 == 'pos':
        return 1
    else:
        return 0

df['分类'] = df['情感分类'].apply(demo)

data = list(df['fenci'])
target = list(df['分类'])


vocab_size = 5000
embedding_dim = 100  # 嵌入向量维度
# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 创建一个tokenizer并用训练数据拟合
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

# 将文本转换为整数序列
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

num_classes = 2
y_train = to_categorical(np.array(y_train), num_classes=num_classes)
y_test = to_categorical(np.array(y_test), num_classes=num_classes)

# 由于每条评论长度不完全一样，我们使用pad_sequences来将评论填充到相同的长度，这里我们设置最大长度为500
maxlen =500
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# 构建CNN模型
model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(units=64, activation='relu'),
    # 修改这一行为两个输出单元和softmax激活函数
    Dense(units=num_classes, activation='softmax')
])

# 你也需要修改编译部分的损失函数
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 早停机制设置
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',     # 监控验证集损失
    patience=10,            # 10个周期内无改进则停止
    restore_best_weights=True  # 恢复最佳权重
)

# 进行训练
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])


# 预测测试集，返回的是每个类别的概率
y_pred_prob = model.predict(X_test)
# 计算每个点的FPR和TPR，这里假设正类是第二类（索引为1）
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred_prob[:, 1])

# 计算PR曲线的数据点
precision, recall, _ = precision_recall_curve(y_test[:, 1], y_pred_prob[:, 1])

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
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('cnn_ROC曲线.png')
# 绘制PR曲线
plt.figure()
plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall')
plt.legend(loc="lower right")
plt.savefig('cnn_PR曲线.png')

# 可视化训练历史
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(history.history['loss'], label='Train loss')
axes[0].plot(history.history['val_loss'], label='Validation loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation accuracy')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('Model loss accuracy.png')
plt.show()

data = pd.DataFrame()
data['Epoch'] = [x for x in range(len(history.history['loss']))]
data['train_loss'] = history.history['loss']
data['train_acc'] = history.history['accuracy']
data['val_loss'] = history.history['val_loss']
data['val_acc'] = history.history['val_accuracy']
data.to_excel('cnn_Model loss accuracy_data.xlsx', index=False)

# 加载新的数据
new_data_df = pd.read_excel('../test.xlsx')
new_data = list(new_data_df['fenci'])

# apply the same preprocessing steps
new_X = tokenizer.texts_to_sequences(new_data)
new_X = pad_sequences(new_X, maxlen=maxlen)

# 预测
new_predictions = model.predict(new_X)
# 选择概率最大的类别
new_predicted_classes = np.argmax(new_predictions, axis=1)

new_data_df['情感分类'] = new_predicted_classes

def demo2(x):
    x1 = int(x)
    if x1 == 1:
        return 'pos'
    else:
        return 'neg'

new_data_df['情感分类'] = new_data_df['情感分类'].apply(demo2)

new_df = new_data_df['情感分类'].value_counts()
x_data = [x for x in new_df.index]
y_data = [y for y in new_df.values]
plt.figure(figsize=(16, 9), dpi=500)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
plt.title('情感占比分布')
plt.tight_layout()
# 添加图例
plt.legend(x_data, loc='lower right')
plt.savefig('情感占比分布.png')
new_data_df.to_excel('cnn_test.xlsx',index=False)
