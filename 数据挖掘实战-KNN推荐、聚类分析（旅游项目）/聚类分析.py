import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import re
import jieba
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn import metrics
import os
from sklearn.metrics import silhouette_score
from IPython.display import Image
import stylecloud


sns.set_style(style="whitegrid")


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


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def fx():
    #从这里开始，首先创建一个文本
    f = open('./非负/fenci.txt', 'w', encoding='utf-8-sig')
    #接着开始读取数据
    for line in new_df['分词']:
        line = line.strip('\n')
        # 停用词过滤
        line = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", line)
        line = re.sub('(\[.*?\])', "", line)
        line = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', line)
        line = re.sub(r'\n', '', line)
        seg_list = jieba.cut(line, cut_all=False)
        cut_words = (" ".join(seg_list))

        # 计算关键词
        all_words = cut_words.split()
        c = Counter()
        for x in all_words:
            if len(x) >= 2 and x != '\r\n' and x != '\n':
                if is_all_chinese(x) == True and x not in stop_words:
                    c[x] += 1
        # Top30
        output = ""
        # print('\n词频统计结果：')
        for (k, v) in c.most_common(30):
            # print("%s:%d"%(k,v))
            output += k + " "

        f.write(output + "\n")
    else:
        f.close()


def kmeans():
    corpus = []

    # 读取预料 一行预料为一个文档
    for line in open('./非负/fenci.txt', 'r', encoding='utf-8-sig').readlines():
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

    silhouette_scores = []
    best_k = None
    # 设置聚类数量K的范围
    range_n_clusters = range(2, 6)
    # 计算每个K值对应的轮廓系数
    for n_clusters in range_n_clusters:
        # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        # labels = kmeans.fit_predict(X)
        # cluster_labels = clusterer.fit_predict(weight)

        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(weight)
        score = silhouette_score(weight, labels)
        silhouette_scores.append(score)

        if best_k is None or score > silhouette_scores[best_k - 2]:
            best_k = n_clusters

    # Print the best K value and its corresponding silhouette score
    print(f"Best K value: {best_k}")
    print(f"Silhouette score for best K value: {silhouette_scores[best_k - 2]}")
        # silhouette_avg = silhouette_score(weight, cluster_labels)
        # silhouette_scores.append(silhouette_avg)

    # 绘制轮廓系数图
    plt.plot(range_n_clusters, silhouette_scores, 'bo-', alpha=0.8)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for K-means clustering')
    plt.savefig('./非负/轮廓系数图.png')
    plt.show()

    n_clusters = best_k
    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))
    # 第二步 聚类Kmeans
    print('Start Kmeans:')

    clf = KMeans(n_clusters=n_clusters, random_state=111)
    pre = clf.fit_predict(weight)

    new_df['聚类结果'] = list(pre)
    new_df.to_csv('./非负/聚类结果.csv', encoding="utf-8-sig",index=False)


def Cluster_analysis(number=None):
    df = pd.read_csv('./非负/聚类结果.csv')
    new_df = df[df['聚类结果'] == number]
    list1 = []
    for n in new_df['分词']:
        n = str(n).split(" ")
        for i in n:
            list1.append(i)

    stylecloud.gen_stylecloud(text=' '.join(list1), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-user',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./非负/聚类{}-词云图.png'.format(number))
    Image(filename='./非负/聚类{}-词云图.png'.format(number))


    counts = {}
    for t in list1:
        counts[t] = counts.get(t, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[1:21]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./非负/top20_聚类{}.csv'.format(number), encoding="utf-8-sig", index=False)


if __name__ == '__main__':
    # fx()
    # kmeans()
    for k in [0,1,2,3,4]:
        Cluster_analysis(k)