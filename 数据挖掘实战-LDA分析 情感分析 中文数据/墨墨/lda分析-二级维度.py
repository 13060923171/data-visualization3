import pandas as pd
import numpy as np
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # 使用Agg后端

import re
import os
import itertools
from tqdm import tqdm
from collections import Counter

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pyLDAvis
import pyLDAvis.gensim 

import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel


# LDA建模
def lda(df,name):
    train = []
    stop_word = []
    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_word.append(line.strip())
    for line in df['fenci']:
        line = [str(word).strip(' ') for word in line.split(' ') if len(word) >= 2 and word not in stop_word]
        train.append(line)

    # 构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    if not os.path.exists("./墨墨-lda"):
        os.mkdir("./墨墨-lda")

    num_topics = 4
    # LDA可视化模块
    # 构建lda主题参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111,
                                          iterations=400)
    # 读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    # 把数据进行可视化处理
    pyLDAvis.save_html(data1, f'./墨墨-lda/墨墨-lda(二级维度-{name}).html')

    # 主题判断模块
    list3 = []
    list2 = []
    # 这里进行lda主题判断
    for i in lda.get_document_topics(corpus)[:]:
        listj = []
        list1 = []
        for j in i:
            list1.append(j)
            listj.append(j[1])
        list3.append(list1)
        bz = listj.index(max(listj))
        list2.append(i[bz][0])

    df['二级-主题概率'] = list3
    df['二级-主题类型'] = list2

    df.to_excel(f'./墨墨-lda/墨墨-lda_data(二级维度-{name}).xlsx', index=False)

    data = df
    # 获取对应主题出现的频次
    new_data = data['二级-主题类型'].value_counts()
    new_data = new_data.sort_index(ascending=True)
    y_data1 = [y for y in new_data.values]

    # 主题词模块
    word = lda.print_topics(num_words=30)
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

    list_Attention = []
    for k in list_gailv1:
        list_k = [float(k1) for k1 in k]
        Attention = sum(list_k) / len(list_k)
        list_Attention.append(Attention)


    y_data2 = []
    for y in y_data1:
        number = float(y / sum(y_data1))
        y_data2.append(float('{:0.5}'.format(number)))

    df1 = pd.DataFrame()
    df1['所属主题'] = topic
    df1['文本数量'] = y_data1
    df1['特征词'] = quanzhong
    df1['维度关注度'] = list_Attention
    df1['关注度内部占比'] = y_data2
    df1.to_excel(f'./墨墨-lda/特征维度(二级维度-{name}).xlsx', index=False)


if __name__ == '__main__':
    df = pd.read_excel('./墨墨-lda/墨墨-lda_data(一级维度).xlsx')
    df['一级-主题类型'] = df['一级-主题类型'].replace(0,'面向教育').replace(1,'技术设计')
    list_number = ['面向教育','技术设计']
    for n in list_number:
        data = df[df['一级-主题类型'] == n]
        lda(data,n)