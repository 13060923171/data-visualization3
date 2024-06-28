import pandas as pd
import re
import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def kmeans(df):
    corpus = []
    for i in df['fenci']:
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

    if not os.path.exists("./聚类/"):
        os.mkdir("./聚类/")

    # silhouette_scores = []
    # best_k = None
    # # 设置聚类数量K的范围
    # range_n_clusters = range(2, 11)
    # # 计算每个K值对应的轮廓系数
    # for n_clusters in range_n_clusters:
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=111)
    #     labels = kmeans.fit_predict(weight)
    #     score = silhouette_score(weight, labels)
    #     silhouette_scores.append(score)
    #
    #     if best_k is None or score > silhouette_scores[best_k - 2]:
    #         best_k = n_clusters


    # # Print the best K value and its corresponding silhouette score
    # print(f"Best K value: {best_k}")
    # print(f"Silhouette score for best K value: {silhouette_scores[best_k - 2]}")



    # data = pd.DataFrame()
    # data['聚类数量'] = range_n_clusters
    # data['轮廓系数'] = silhouette_scores
    # data.to_csv('./聚类/轮廓系数.csv', encoding="utf-8-sig",index=False)
    # # 绘制轮廓系数图
    # plt.plot(range_n_clusters, silhouette_scores, 'bo-', alpha=0.8)
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette score')
    # plt.title('Silhouette score for K-means clustering')
    # plt.savefig('./聚类/轮廓系数图.png')
    # plt.show()

    n_clusters = 3

    clf = KMeans(n_clusters=n_clusters, random_state=111)
    pre = clf.fit_predict(weight)

    df['聚类结果'] = pre
    df.to_csv('./聚类/聚类结果.csv', encoding="utf-8-sig",index=False)

    pca = PCA(n_components=2)  # 输出两维
    newData = pca.fit_transform(weight)  # 载入N维

    x = [n[0] for n in newData]
    y = [n[1] for n in newData]
    plt.figure(figsize=(9,6),dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(x, y, c=pre, s=100)
    plt.title("聚类图")
    plt.savefig('./聚类/聚类图.jpg')
    plt.show()

    # # 获取每个类的聚类相关词汇
    # centroids = clf.cluster_centers_  # 获取聚类中心
    # order_centroids = centroids.argsort()[:, ::-1]  # 获取聚类中心的排序
    # terms = vectorizer.get_feature_names_out()  # 获取所有词汇
    # cluster_keywords = []
    # cluster_weights = []
    # for i in range(n_clusters):
    #     top_words_indices = order_centroids[i, :30]
    #     top_words = [terms[ind] for ind in top_words_indices]
    #     weights = [clf.cluster_centers_[i, ind] for ind in top_words_indices]
    #     cluster_keywords.append(top_words)  # 每个类别选择前30个作为聚类相关词汇
    #     cluster_weights.append(weights)  # 对应的权重


    # x_data = []
    # y_data = []
    # z_data = []
    # len1 = [int(i) for i in range(len(cluster_weights))]
    # # 打印每个类的聚类相关词汇
    # for i,keyword,weight in zip(len1,cluster_keywords,cluster_weights):
    #     weight1 = [str(round(w,2)) for w in weight]
    #     x1 = 'Cluster :' + str(i + 1)
    #     dic = {}
    #     for k,w in zip(keyword,weight1):
    #         dic[k] = w
    #     y_data.append(dic)
    #     x_data.append(x1)
    #
    # data = pd.DataFrame()
    # data['Cluster'] = x_data
    # data['keyword'] = y_data
    # data.to_csv('./聚类/聚类相关词汇.csv',encoding='utf-8-sig',index=False)


if __name__ == '__main__':
    data1 = pd.read_csv('data1.csv')
    kmeans(data1)



