import jieba
import pandas as pd
import numpy as np
import jieba.posseg as posseg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
import seaborn as sns
sns.set_style(style="whitegrid")


def tf_idf():
    # 判断是否为中文
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    # 导入停用词列表
    stop_words = ['满意','不满意']
    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    def demo(content_series):
        # 对文本进行分词和词性标注
        words = posseg.cut(content_series)
        # 保存名词和动名词的列表
        nouns_and_adjs = []
        # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
        for word, flag in words:
            if flag == 'an' or flag == 'vn':
                if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                    # 如果是名词和动名词，就将其保存到列表中
                    nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN

    df1 = pd.read_excel('data.xlsx')
    df2 = pd.read_excel('data1.xlsx')
    df = pd.concat([df1,df2],axis=0)
    df['new_content'] = df['帖子正文'].apply(demo)
    df = df.dropna(how='any',axis=0)

    corpus = []
    for i in df['new_content']:
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
    df2 = df2.sort_values(by=['tfidf'], ascending=False)
    df2 = df2.iloc[:50]
    x_data = list(df2['word'])[:50]
    y_data = list(df2['tfidf'])[:50]
    x_data.reverse()
    y_data.reverse()
    plt.figure(figsize=(12, 9), dpi=500)
    plt.barh(x_data, y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("TF-IDF TOP30")
    plt.xlabel("value")
    plt.savefig('TFIDF.png')
    plt.show()
    df2.to_csv('评论二级特征的 TF-IDF 值.csv', encoding='utf-8-sig', index=False)

    k_mean(df,x_data)


def k_mean(df,x_data):
    corpus = []
    def demo(x):
        x2 = []
        x1 = str(x).split(" ")
        for i in x1:
            if i in x_data:
                x2.append(i)
        if len(x2) != 0:
            return ' '.join(x2)
        else:
            return np.NAN

    df['new_content'] = df['new_content'].apply(demo)
    df = df.dropna(subset=['new_content'],axis=0)

    for i in df['new_content']:
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

    silhouette_scores = []
    best_k = None
    # 设置聚类数量K的范围
    range_n_clusters = range(2, 16)
    # 计算每个K值对应的轮廓系数
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=111)
        labels = kmeans.fit_predict(weight)
        score = silhouette_score(weight, labels)
        silhouette_scores.append(score)

        if best_k is None or score > silhouette_scores[best_k - 2]:
            best_k = n_clusters

    # Print the best K value and its corresponding silhouette score
    print(f"Best K value: {best_k}")
    print(f"Silhouette score for best K value: {silhouette_scores[best_k - 2]}")

    data = pd.DataFrame()
    data['聚类数量'] = range_n_clusters
    data['轮廓系数'] = silhouette_scores
    data.to_csv('轮廓系数.csv', encoding="utf-8-sig", index=False)
    # 绘制轮廓系数图
    plt.figure(figsize=(12, 9), dpi=500)
    plt.plot(range_n_clusters, silhouette_scores, 'bo-', alpha=0.8)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for K-means clustering')
    plt.savefig('轮廓系数图.png')
    plt.show()
    for i in range(2,16):
        n_clusters = i
        # 打印特征向量文本内容
        print('Features length: ' + str(len(word)))
        # 第二步 聚类Kmeans
        print('Start Kmeans:')

        clf = KMeans(n_clusters=n_clusters, random_state=111)
        pre = clf.fit_predict(weight)

        # 中心点
        print(clf.cluster_centers_)
        print(clf.inertia_)
        # 第三步 图形输出 降维
        pca = PCA(n_components=n_clusters)  # 输出两维
        newData = pca.fit_transform(weight)  # 载入N维

        x = [n[0] for n in newData]
        y = [n[1] for n in newData]
        plt.figure(figsize=(16, 9), dpi=500)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False
        plt.scatter(x, y, c=pre, s=100)
        plt.savefig('./聚类图/k={}-词性聚类图.png'.format(i))
        plt.show()

    #
    # n_clusters = 2
    # # 打印特征向量文本内容
    # print('Features length: ' + str(len(word)))
    # # 第二步 聚类Kmeans
    # print('Start Kmeans:')
    #
    # clf = KMeans(n_clusters=n_clusters, random_state=111)
    # pre = clf.fit_predict(weight)
    #
    # # 中心点
    # print(clf.cluster_centers_)
    # print(clf.inertia_)
    # # 第三步 图形输出 降维
    # pca = PCA(n_components=n_clusters)  # 输出两维
    # newData = pca.fit_transform(weight)  # 载入N维
    #
    # x = [n[0] for n in newData]
    # y = [n[1] for n in newData]
    # plt.figure(figsize=(16, 9), dpi=500)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.scatter(x, y, c=pre, s=100)
    # plt.savefig('词性聚类图.png')
    # plt.show()




if __name__ == '__main__':
    tf_idf()