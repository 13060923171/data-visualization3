import keras
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

df = pd.read_excel("train.xlsx")
data = list(df['分词'])
target = list(df['情感极性（好1，中0，差-1）'])


vocab_size = 5000
# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 创建一个tokenizer并用训练数据拟合
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

# 将文本转换为整数序列
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

num_classes = 3
y_train = to_categorical(np.array(y_train), num_classes=num_classes)
y_test = to_categorical(np.array(y_test), num_classes=num_classes)

# 由于每条评论长度不完全一样，我们使用pad_sequences来将评论填充到相同的长度，这里我们设置最大长度为500
maxlen =500
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=maxlen))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# 实例化一个EarlyStopping回调并添加到model的训练中
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# # 保存在验证集上性能最好的模型
# checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# 保存模型历史记录以便于后续的可视化
history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stop])

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

# 加载新的数据
new_data_df = pd.read_excel('test.xlsx')
new_data = list(new_data_df['分词'])

# apply the same preprocessing steps
new_X = tokenizer.texts_to_sequences(new_data)
new_X = pad_sequences(new_X, maxlen=maxlen)

# 预测
new_predictions = model.predict(new_X)
# 选择概率最大的类别
new_predicted_classes = np.argmax(new_predictions, axis=1)

new_data_df['情感极性（好1，中0，差-1）'] = new_predicted_classes

def process(x):
    x1 = int(x)
    if x1 == 2:
        return '-1'
    else:
        return x1

new_data_df['情感极性（好1，中0，差-1）'] = new_data_df['情感极性（好1，中0，差-1）'].apply(process)
new_data_df.to_excel('new_test.xlsx',index=False)

