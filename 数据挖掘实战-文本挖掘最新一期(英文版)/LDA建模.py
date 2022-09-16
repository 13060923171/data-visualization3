from gensim import corpora, models
import pyLDAvis.gensim
import pyLDAvis.sklearn
import nltk
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('new_data.csv')

stop_words = []

with open("常用英文停用词(NLP处理英文必备)stopwords.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


def tokenize_only(text):  # 分词器，仅分词
    # 首先分句，接着分词，而标点也会作为词例存在
    try:
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # 过滤所有不含字母的词例（例如：数字、纯标点）
        if len(tokens) != 0:
            for token in tokens:
                token = str(token)
                if re.search('[a-zA-Z]', token):
                    filtered_tokens.append(token)
            if len(filtered_tokens) >= 5:
                return ' '.join(filtered_tokens)
        else:
            return ' '
    except:
        return ' '


df['英文翻译'].astype('str')
df['英文翻译1'] = df['英文翻译'].apply(tokenize_only)
df = df.dropna(subset=['英文翻译1'],axis=0)
f = open('class-fenci.txt', 'w', encoding='utf-8')
for d in df['英文翻译1']:
    c = Counter()
    tokens = nltk.word_tokenize(d)
    tagged = nltk.pos_tag(tokens)
    for t in tagged:
        if t[0] not in stop_words and t[0] != '\r\n' and len(t[0]) > 1:
            c[t[0]] += 1
    # Top30
    output = ""
    for (k, v) in c.most_common(50):
        # print("%s:%d"%(k,v))
        output += k + " "
    f.write(output + "\n")

else:
    f.close()


fr = open('class-fenci.txt', 'r', encoding='utf-8')
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
plt.title('LDA评论主题数寻优')
plt.xlabel('主题数')
plt.ylabel('平均余弦相似度')
plt.savefig('LDA评论主题数寻优.png')
plt.show()

corpus = []
# 读取预料 一行预料为一个文档
for line in open('class-fenci.txt', 'r', encoding='utf-8-sig').readlines():
    corpus.append(line.strip())

# 设置特征数
n_features = 2000
tf_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                max_df=0.99,
                                min_df=0.002)  # 去除文档内出现几率过大或过小的词汇

tf = tf_vectorizer.fit_transform(corpus)

n_topics = 2
lda = LatentDirichletAllocation(n_components=n_topics,
                                max_iter=100,
                                learning_method='online',
                                learning_offset=50,
                                random_state=0)
lda.fit(tf)
data = pyLDAvis.sklearn.prepare(lda, tf,tf_vectorizer)
pyLDAvis.save_html(data, 'lda.html')
