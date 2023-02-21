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
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def kmeans():
    df = pd.read_csv('./data/new_data.csv')
    corpus = []

    # 读取预料 一行预料为一个文档
    for line in open('./data/fenci.txt', 'r', encoding='utf-8-sig').readlines():
        corpus.append(line.strip())

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()

    # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    n_clusters = 4
    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))
    # 第二步 聚类Kmeans
    print('Start Kmeans:')

    clf = KMeans(n_clusters=n_clusters)
    pre = clf.fit_predict(weight)

    result = pd.concat((df, pd.DataFrame(pre)), axis=1)
    result.rename({0: '聚类结果'}, axis=1, inplace=True)
    result.to_csv('./data/聚类结果.csv', encoding="utf-8-sig")

    # 中心点
    print(clf.cluster_centers_)
    print(clf.inertia_)

    #图形输出 降维

    pca = PCA(n_components=n_clusters)  # 输出两维
    newData = pca.fit_transform(weight)  # 载入N维

    x = [n[0] for n in newData]
    y = [n[1] for n in newData]
    plt.figure(figsize=(12, 9), dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(x, y, c=pre, s=100)
    plt.title("聚类分析图")
    plt.savefig('./data/聚类分析图.png')
    plt.show()


if __name__ == '__main__':
    kmeans()