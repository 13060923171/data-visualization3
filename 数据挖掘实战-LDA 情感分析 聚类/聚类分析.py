import pandas as pd
import re
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score


sns.set_style(style="whitegrid")


def kmeans():
    df = pd.read_csv('new_data.csv')

    stop_word = ["不错", "手表", "收到", "儿童", "宝贝", "孩子", "手机", "喜欢", "真的", "购物", "购买"]

    corpus = []

    for i in df['分词']:
        corpus.append(i.strip(' '))

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

    # silhouette_scores = []
    # best_k = None
    # # 设置聚类数量K的范围
    # range_n_clusters = range(2, 15)
    # # 计算每个K值对应的轮廓系数
    # for n_clusters in range_n_clusters:
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=111)
    #     labels = kmeans.fit_predict(weight)
    #     score = silhouette_score(weight, labels)
    #     silhouette_scores.append(score)
    #
    #     if best_k is None or score > silhouette_scores[best_k - 2]:
    #         best_k = n_clusters
    #
    # # Print the best K value and its corresponding silhouette score
    # print(f"Best K value: {best_k}")
    # print(f"Silhouette score for best K value: {silhouette_scores[best_k - 2]}")
    #
    # data = pd.DataFrame()
    # data['聚类数量'] = range_n_clusters
    # data['轮廓系数'] = silhouette_scores
    # data.to_csv('轮廓系数.csv', encoding="utf-8-sig",index=False)
    # # 绘制轮廓系数图
    # plt.plot(range_n_clusters, silhouette_scores, 'bo-', alpha=0.8)
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette score')
    # plt.title('Silhouette score for K-means clustering')
    # plt.savefig('轮廓系数图.png')
    # plt.show()

    n_clusters = 4
    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))
    # 第二步 聚类Kmeans
    print('Start Kmeans:')

    clf = KMeans(n_clusters=n_clusters, random_state=111)
    pre = clf.fit_predict(weight)

    # 获取每个类的聚类相关词汇
    centroids = clf.cluster_centers_  # 获取聚类中心
    order_centroids = centroids.argsort()[:, ::-1]  # 获取聚类中心的排序
    terms = vectorizer.get_feature_names_out()  # 获取所有词汇
    cluster_keywords = []
    for i in range(n_clusters):
        cluster_keywords.append([terms[ind] for ind in order_centroids[i, :20]])  # 每个类别选择前6个作为聚类相关词汇

    x_data = []
    y_data = []
    # 打印每个类的聚类相关词汇
    for i, keywords in enumerate(cluster_keywords):
        x1 = 'Cluster :' + str(i + 1)
        keywords1 = [x for x in keywords if x not in stop_word]
        y1 = ", ".join(keywords1)
        x_data.append(x1)
        y_data.append(y1)

    data = pd.DataFrame()
    data['Cluster'] = x_data
    data['keyword'] = y_data
    data.to_csv('聚类相关词汇.csv',encoding='utf-8-sig',index=False)

    # 图形输出 降维
    pca = PCA(n_components=2)  # 输出两维
    newData = pca.fit_transform(weight)  # 载入N维

    x = [n[0] for n in newData]
    y = [n[1] for n in newData]
    plt.figure(figsize=(16, 9), dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(x, y, c=pre, s=100)
    plt.xlabel("PCA Dimension1")
    plt.ylabel("PCA Dimension2")
    plt.title("Cluster analysis")
    plt.savefig('聚类分析图.png')
    plt.show()

    df['聚类结果'] = list(pre)
    df.to_csv('聚类结果.csv', encoding="utf-8-sig",index=False)


if __name__ == '__main__':
    kmeans()


