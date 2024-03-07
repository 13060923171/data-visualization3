from hmmlearn import hmm
import numpy as np
import re
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter


# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


clean_dict = {}
with open('文本语料库.txt', 'r', encoding='utf-8-sig') as f:
    corpus = f.readlines()
for line in corpus:
    items = line.split()
    for item in items:
        parts = item.rsplit('/', 1)
        if len(parts) > 1:
            key, value = parts[0], parts[1]
            if key not in clean_dict and is_all_chinese(key) == True:
                clean_dict[key] = value

word_dict = {}
for key, value in clean_dict.items():
    if value not in word_dict.values():
        word_dict[key] = value

train_set = []
for key, values in word_dict.items():
    word = str(key) + ":" + str(values)
    train_set.append(word)


def main1(content, fenci):
    list_true = []
    test_data = []
    test_set = []
    word = str(fenci).split(" ")
    for w in word:
        test_data.append(w)

    test_data = list(set(test_data))
    for w in test_data:
        true_flag = word_dict.get(w)
        if true_flag is not None:
            list_true.append(true_flag)
            test_set.append(w)

    # 分别获取词和词性
    words, tags = zip(*[item.split(':') for item in train_set])

    # 对于词和词性的编码
    unique_words = list(set(words))
    word2index = {word: i for i, word in enumerate(unique_words)}

    unique_tags = list(set(tags))
    tag2index = {tag: i for i, tag in enumerate(unique_tags)}

    # 编码后的观测数据
    obs = [word2index[word] for word in test_set]

    # 调整平滑参数
    smoothing_value = 0.001
    emission_probability = np.ones((len(unique_tags), len(unique_words))) * smoothing_value
    transition_probability = np.ones((len(unique_tags), len(unique_tags))) * smoothing_value
    # Generate bigram and trigram for word and tags
    word_pairs = [(words[i], tags[i]) for i in range(len(tags) - 1)]
    word_triplets = [(words[i], tags[i], tags[i + 1]) for i in range(len(tags) - 2)]
    # Count occurrences for transition and emission
    word_count = Counter(words)
    tag_count = Counter(tags)
    word_pair_count = Counter(word_pairs)
    word_triplet_count = Counter(word_triplets)

    # Update transition and emission probability with smoothing
    for i, tag in enumerate(unique_tags):
        for j, word in enumerate(unique_words):
            emission_probability[i, j] = (word_pair_count[(word, tag)] + smoothing_value) / (
                    tag_count[tag] + smoothing_value * len(unique_words))
        for k, next_tag in enumerate(unique_tags):
            transition_probability[i, k] = (word_triplet_count[(tag, word, next_tag)] + smoothing_value) / (
                    word_pair_count[(tag, word)] + smoothing_value * len(unique_tags))
    # 实例化和训练 HMM 模型
    model = hmm.MultinomialHMM(n_components=len(unique_tags), n_iter=100, params='ste', init_params='ste')
    model.startprob_ = np.ones(len(unique_tags)) / len(unique_tags)
    # Update model training part with transition and emission probability
    model.transmat_ = transition_probability / transition_probability.sum(axis=1, keepdims=True)
    model.emissionprob_ = emission_probability / emission_probability.sum(axis=1, keepdims=True)
    try:
        # 训练模型
        model = model.fit(np.array(obs).reshape(-1, 1))
        # 预测
        logprob, seq = model.decode(np.array(obs).reshape(-1, 1))
        # 将预测结果转化为标签
        predicted_tags = [unique_tags[index] for index in seq]
        predicted_words = [unique_words[index] for index in seq]
        predicted_words = " ".join(predicted_words)
        hmm_prec, hmm_rec, hmm_f_score, _ = precision_recall_fscore_support(list_true, predicted_tags,
                                                                            average='weighted')

        data = pd.DataFrame()
        data['评论内容'] = [content]
        data['fenci'] = [predicted_words]
        data['Precision'] = ['{:.2f}'.format(hmm_prec)]
        data['Recall'] = ['{:.2f}'.format(hmm_rec)]
        data['FScore'] = ['{:.2f}'.format(hmm_f_score)]
        data.to_csv('hmm_data.csv', index=False, encoding='utf-8-sig', header=False, mode='a+')
    except:
        pass


df = pd.read_csv('data.csv')
data = pd.DataFrame()
data['评论内容'] = ['评论内容']
data['fenci'] = ['fenci']
data['Precision'] = ['Precision']
data['Recall'] = ['Recall']
data['FScore'] = ['FScore']
data.to_csv('hmm_data.csv', index=False, encoding='utf-8-sig', header=False, mode='w')

for c, f in zip(df['评论内容'], df['fenci']):
    main1(c, f)


def main2():
    df = pd.read_csv('hmm_data.csv')
    hmm_prec = df['Precision'].mean()
    hmm_rec = df['Recall'].mean()
    hmm_f_score = df['FScore'].mean()

    data = pd.DataFrame()
    data['Precision'] = ['{:.2f}'.format(hmm_prec)]
    data['Recall'] = ['{:.2f}'.format(hmm_rec)]
    data['FScore'] = ['{:.2f}'.format(hmm_f_score)]
    data.to_csv('prec_hmm.csv', index=False, encoding='utf-8-sig')

    df = df.drop(['Precision', 'Recall', 'FScore'], axis=1)
    df.to_csv('hmm_data.csv', index=False, encoding='utf-8-sig')


main2()