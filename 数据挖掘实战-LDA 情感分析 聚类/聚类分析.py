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
        word_list = i.split()  # 我们首先将字符串分割成词的列表
        filtered_i = ' '.join(
            [word for word in word_list if word not in stop_word])  # 对于 i 中的每一个词，如果它不在 stop_word 列表中，则保留，并重新组合成字符串
        corpus.append(filtered_i)

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
    cluster_weights = []
    for i in range(n_clusters):
        top_words_indices = order_centroids[i, :20]
        top_words = [terms[ind] for ind in top_words_indices]
        weights = [clf.cluster_centers_[i, ind] for ind in top_words_indices]
        cluster_keywords.append(top_words)  # 每个类别选择前20个作为聚类相关词汇
        cluster_weights.append(weights)  # 对应的权重


    x_data = []
    y_data = []
    z_data = []
    len1 = [int(i) for i in range(len(cluster_weights))]
    # 打印每个类的聚类相关词汇
    for i,keyword,weight in zip(len1,cluster_keywords,cluster_weights):
        weight1 = [str(round(w,2)) for w in weight]
        x1 = 'Cluster :' + str(i + 1)
        dic = {}
        for k,w in zip(keyword,weight1):
            dic[k] = w
        y_data.append(dic)
        x_data.append(x1)

    data = pd.DataFrame()
    data['Cluster'] = x_data
    data['keyword'] = y_data
    data.to_csv('聚类相关词汇.csv',encoding='utf-8-sig',index=False)

    # # 图形输出 降维
    # pca = PCA(n_components=2)  # 输出两维
    # newData = pca.fit_transform(weight)  # 载入N维
    #
    # x = [n[0] for n in newData]
    # y = [n[1] for n in newData]
    # plt.figure(figsize=(16, 9), dpi=300)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.scatter(x, y, c=pre, s=100)
    # plt.xlabel("PCA Dimension1")
    # plt.ylabel("PCA Dimension2")
    # plt.title("Cluster analysis")
    # plt.savefig('聚类分析图.png')
    # plt.show()
    #
    # df['聚类结果'] = list(pre)
    # df.to_csv('聚类结果.csv', encoding="utf-8-sig",index=False)


if __name__ == '__main__':
    kmeans()


