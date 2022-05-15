from gensim import corpora, models, similarities
import pyLDAvis
import pyLDAvis.gensim
import pandas as pd
import nltk
import re
from collections import Counter
from nltk.stem.snowball import SnowballStemmer  # 返回词语的原型，去掉ing等
import itertools
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import os


df = pd.read_csv('new_kw2.csv')

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

stop_words = []

with open("常用英文停用词(NLP处理英文必备)stopwords.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


def tokenize_only(text):  # 分词器，仅分词
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return ' '.join(filtered_tokens)


df['comment_title_new'] = df['comment_title_new'].apply(tokenize_only)
df['comment_title_new'].astype('str')
f = open('kw2-class-fenci.txt', 'w', encoding='utf-8')
for d in df['comment_title_new']:
    c = Counter()
    tokens = nltk.word_tokenize(d)
    tagged = nltk.pos_tag(tokens)
    for t in tagged:
        if t[0] not in stop_words and t[0] != '\r\n' and len(t[0]) > 1:
            c[t[0]] += 1
    # Top20
    output = ""
    for (k, v) in c.most_common():
        # print("%s:%d"%(k,v))
        output += k + " "
    f.write(output + "\n")

else:
    f.close()


fr = open('kw2-class-fenci.txt', 'r', encoding='utf-8')
train = []
for line in fr.readlines():
    line = [word.strip() for word in line.split(' ')]
    train.append(line)

dictionary = corpora.Dictionary(train)
corpus = [dictionary.doc2bow(text) for text in train]

# 构造主题数寻优函数
def cos(vector1, vector2):  # 余弦相似度函数
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return (None)
    else:
        return (dot_product / ((normA * normB) ** 0.5))

    # 主题数寻优

def lda_k(x_corpus, x_dict):
    # 初始化平均余弦相似度
    mean_similarity = []
    mean_similarity.append(1)

    # 循环生成主题并计算主题间相似度
    for i in np.arange(2, 11):
        lda = models.LdaModel(x_corpus, num_topics=i, id2word=x_dict)  # LDA模型训练
        for j in np.arange(i):
            term = lda.show_topics(num_words=50)

        # 提取各主题词
        top_word = []
        for k in np.arange(i):
            top_word.append([''.join(re.findall('"(.*)"', i)) \
                             for i in term[k][1].split('+')])  # 列出所有词

        # 构造词频向量
        word = sum(top_word, [])  # 列出所有的词
        unique_word = set(word)  # 去除重复的词

        # 构造主题词列表，行表示主题号，列表示各主题词
        mat = []
        for j in np.arange(i):
            top_w = top_word[j]
            mat.append(tuple([top_w.count(k) for k in unique_word]))

        p = list(itertools.permutations(list(np.arange(i)), 2))
        l = len(p)
        top_similarity = [0]
        for w in np.arange(l):
            vector1 = mat[p[w][0]]
            vector2 = mat[p[w][1]]
            top_similarity.append(cos(vector1, vector2))

        # 计算平均余弦相似度
        mean_similarity.append(sum(top_similarity) / l)
    return (mean_similarity)

# 计算主题平均余弦相似度
word_k = lda_k(corpus, dictionary)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 8), dpi=300)
plt.plot(word_k)
plt.title('kw2-LDA评论主题数寻优')
plt.xlabel('主题数')
plt.ylabel('平均余弦相似度')
plt.savefig('kw2-LDA评论主题数寻优.png')
plt.show()

topic_lda = word_k.index(min(word_k)) + 1

lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_lda)

data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(data, 'kw2-lda.html')