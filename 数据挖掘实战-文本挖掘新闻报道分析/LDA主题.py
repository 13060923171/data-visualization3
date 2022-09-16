from gensim import corpora, models, similarities
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.sklearn
import nltk
import re
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.read_excel('./data/爱尔兰日报.xlsx')


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


df['new_内容'] = df['new_内容'].apply(tokenize_only)
df['new_内容'].astype('str')
f = open('./data/爱尔兰-class-fenci.txt', 'w', encoding='utf-8-sig')
for d in df['new_内容']:
    c = Counter()
    tokens = nltk.word_tokenize(d)
    tagged = nltk.pos_tag(tokens)
    for t in tagged:
        if t[0] not in stop_words and t[0] != '\r\n' and len(t[0]) > 2:
            c[t[0]] += 1
    # Top30
    output = ""
    for (k, v) in c.most_common(30):
        # print("%s:%d"%(k,v))
        output += k + " "
    f.write(output + "\n")

else:
    f.close()


fr = open('./data/爱尔兰-class-fenci.txt', 'r', encoding='utf-8-sig')
train = []
for line in fr.readlines():
    line = [word.strip() for word in line.split(' ') if len(word) > 2]
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
plt.title('爱尔兰-LDA评论主题数寻优')
plt.xlabel('主题数')
plt.ylabel('平均余弦相似度')
plt.savefig('./data/爱尔兰-LDA评论主题数寻优.png')
plt.show()


topic_lda = 4
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_lda)
data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(data, './data/爱尔兰-lda.html')
list3 = []
list2 = []
for i in lda.get_document_topics(corpus)[:]:
    listj = []
    list1 = []
    for j in i:
        list1.append(j)
        listj.append(j[1])
    list3.append(list1)
    bz = listj.index(max(listj))
    # print(i[bz][0])
    list2.append(i[bz][0])

df['主题概率'] = list3
df['主题类型'] = list2
df.to_csv('./data/new_爱尔兰.csv',encoding='utf-8-sig',index=False)




# corpus = []
# # 读取预料 一行预料为一个文档
# for line in open('./data/爱尔兰-class-fenci.txt', 'r', encoding='utf-8-sig').readlines():
#     corpus.append(line.strip())
#
# # -----------------------  第二步 计算TF-IDF值  -----------------------
# # 设置特征数
# n_features = 2000
# tf_vectorizer = TfidfVectorizer(strip_accents='unicode',
#                                 max_features=n_features,
#                                 max_df=0.99,
#                                 min_df=0.002)  # 去除文档内出现几率过大或过小的词汇
#
# tf = tf_vectorizer.fit_transform(corpus)
#
#
# lda1 = LatentDirichletAllocation(n_components=topic_lda,
#                                  max_iter=100,
#                                  learning_method='online',
#                                  learning_offset=50,
#                                  random_state=0)
# lda1.fit(tf)
# data = pyLDAvis.sklearn.prepare(lda1, tf,tf_vectorizer)
# pyLDAvis.save_html(data, './data/爱尔兰-lda.html')