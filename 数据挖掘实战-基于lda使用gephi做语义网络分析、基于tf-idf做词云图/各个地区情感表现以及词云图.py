import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib

from collections import Counter
import itertools
import jieba
import jieba.posseg as pseg

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread


def tf_idf(df):
    corpus = []
    for i in df['分词']:
        corpus.append(i.strip())

        # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names_out()

    # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    data = {'word': word,
            'tfidf': weight.sum(axis=0).tolist()}

    df2 = pd.DataFrame(data)
    df2['tfidf'] = df2['tfidf'].astype('float64')
    df2 = df2.sort_values(by=['tfidf'],ascending=False)
    df2.to_csv('./爱奇艺/TF-IDF相关数据.xlsx'.format(date_time,area,name),encoding='utf-8-sig',index=False)

    df3 = df2.iloc[:30]
    x_data = list(df3['word'])
    y_data = list(df3['tfidf'])
    x_data.reverse()
    y_data.reverse()
    plt.figure(figsize=(12, 9))
    plt.barh(x_data, y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.title("tf-idf 权重最高的top30词汇")
    plt.xlabel("权重")
    plt.savefig('./爱奇艺/tf-idf top30.png')



if __name__ == '__main__':
    df = pd.read_excel('./爱奇艺/新_评论表.xlsx')
    tf_idf(df)







