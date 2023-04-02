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
from gensim import corpora, models
import itertools
import pyLDAvis
import pyLDAvis.gensim
import gensim
from tqdm import tqdm
import os
from gensim.models import LdaModel


df = pd.read_csv('./data/data_情感分析.csv')
new_df = df[df['emotion_type'] == '非负']
new_df = new_df.dropna(subset=['分词'], axis=0)
if not os.path.exists("./非负"):
    os.mkdir("./非负")

stop_words = []

with open("stopwords_cn.txt", 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


def lda():
    # 读取数据
    fr = open('./非负/fenci.txt', 'r', encoding='utf-8-sig')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ') if len(word) >= 2]
        train.append(line)
    # 构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    # 困惑度模块
    x_data = []
    y_data = []
    z_data = []
    for i in tqdm(range(2, 15)):
        x_data.append(i)
        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, random_state=111, iterations=400)
        # 困惑度计算
        _, perplexity = lda.log_perplexity(corpus)
        y_data.append(perplexity)
        # 一致性计算
        coherencemodel = models.CoherenceModel(model=lda, texts=train, dictionary=dictionary, coherence='c_v')
        z_data.append(coherencemodel.get_coherence())

    # 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 绘制困惑度折线图
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x_data, y_data, marker="o")
    plt.title("perplexity_values")
    plt.xlabel('num topics')
    plt.ylabel('perplexity score')
    # 绘制一致性的折线图
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x_data, z_data, marker="o")
    plt.title("coherence_values")
    plt.xlabel("num topics")
    plt.ylabel("coherence score")

    plt.savefig('./非负/困惑度和一致性.png')
    plt.show()
    # 将上面获取的数据进行保存
    df5 = pd.DataFrame()
    df5['主题数'] = x_data
    df5['困惑度'] = y_data
    df5['一致性'] = z_data
    df5.to_csv('./非负/困惑度和一致性.csv', encoding='utf-8-sig', index=False)
    # num_topics = input('请输入主题数:')
    num_topics = int(z_data.index(max(z_data))+1)
    # LDA可视化模块
    # 构建lda主题参数
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    # 读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    # 把数据进行可视化处理
    pyLDAvis.save_html(data1, './非负/lda.html')

    # 主题词模块
    word = lda.print_topics(num_words=20)
    topic = []
    quanzhong = []
    list_gailv = []
    list_gailv1 = []
    list_word = []
    # 根据其对应的词，来获取其相应的权重
    for w in word:
        ci = str(w[1])
        c1 = re.compile('\*"(.*?)"')
        c2 = c1.findall(ci)
        list_word.append(c2)
        c3 = '、'.join(c2)

        c4 = re.compile(".*?(\d+).*?")
        c5 = c4.findall(ci)
        for c in c5[::1]:
            if c != "0":
                gailv = str(0) + '.' + str(c)
                list_gailv.append(gailv)
        list_gailv1.append(list_gailv)
        list_gailv = []
        zt = "Topic" + str(w[0])
        topic.append(zt)
        quanzhong.append(c3)

    # 把上面权重的词计算好之后，进行保存为csv文件
    df2 = pd.DataFrame()
    for j, k, l in zip(topic, list_gailv1, list_word):
        df2['{}-主题词'.format(j)] = l
        df2['{}-权重'.format(j)] = k
    df2.to_csv('./非负/主题词分布表.csv', encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    lda()