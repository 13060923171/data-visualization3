import pycrfsuite


# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

word_dict = {}
with open('文本语料库.txt','r',encoding='utf-8-sig')as f:
    corpus = f.readlines()
for line in corpus:
    items = line.split()
    for item in items:
        parts = item.rsplit('/', 1)
        if len(parts) > 1:
            key, value = parts[0], parts[1]
            if key not in word_dict:
                word_dict[key] = value
train_data1 = []
for key,value in word_dict.items():
    if is_all_chinese(key) == True:
        train_data1.append((str(key),str(value)))

print(train_data1)
# 准备训练数据
train_data = [
    train_data1
]

# 定义特征函数
def word2features(sent, i):
    word = sent[i][0]

    features = {
        'word': word,
        # 添加更多特征
    }
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

# 转换训练数据为特征格式
X_train = [sent2features(s) for s in train_data]
y_train = [sent2labels(s) for s in train_data]

# 训练 CRF 模型
trainer = pycrfsuite.Trainer(verbose=False)
for x, y in zip(X_train, y_train):
    trainer.append(x, y)
trainer.set_params({
    'c1': 1.0,
    'c2': 1e-3,
    'max_iterations': 1000,  # 增加最大迭代次数，允许模型更多的训练时间
    'feature.possible_transitions': True
})
trainer.train('crf_model.crfsuite')



from hmmlearn import hmm
import numpy as np

# 假设提供的中文数据集
data = [
    ["我", "爱", "自然", "语言", "处理"],
    ["你", "是", "谁"]
]

# 将中文文本转换为数字序列
word2idx = {"<UNK>": 0}
idx2word = {0: "<UNK>"}
for sent in data:
    for word in sent:
        if word not in word2idx:
            idx = len(word2idx)
            word2idx[word] = idx
            idx2word[idx] = word

# 构建观测序列
observations = np.array([[word2idx.get(word, 0) for word in sent] for sent in data])

# 构建 HMM 模型
model = hmm.MultinomialHMM(n_components=2, n_iter=100)

# 训练模型
model.fit(observations)

# 分词和词性标注
for i in range(len(data)):
    X = np.atleast_2d(observations[i]).T
    hidden_states = model.predict(X)
    for j in range(len(data[i])):
        print("分词：{}，词性：{}".format(data[i][j], "state"+str(hidden_states[j])))

