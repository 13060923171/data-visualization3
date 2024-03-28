import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import itertools
import pyLDAvis
import pyLDAvis.gensim
from tqdm import tqdm
import os
from gensim.models import LdaModel
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel


#LDA建模
def lda():
    df = pd.read_csv('new_data.csv')
    # 导入停用词列表
    stop_words = ["感觉","夜熙","真的","天妤","越来越"]
    train = []
    for line in df['分词']:
        line = [word.strip(' ') for word in line.split(' ') if len(word) >= 2 and word not in stop_words]
        train.append(line)

    #构建为字典的格式
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
            lda = gensim.models.ldamodel.LdaModel(x_corpus, num_topics=i, id2word=x_dict)  # LDA模型训练
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
    num_topics = input('请输入主题数:')

    #LDA可视化模块
    #构建lda主题参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, 'lda.html')


if __name__ == '__main__':
    lda()

