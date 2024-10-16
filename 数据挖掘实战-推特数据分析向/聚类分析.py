import pandas as pd
import numpy as np
import re
import time
import nltk
import codecs
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
# from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

df = pd.read_csv('new_推特.csv')
list_number = []
for i in range(0,len(df),20000):
    list_number.append(i)

for d in tqdm(range(len(list_number)-1)):
    sid = SentimentIntensityAnalyzer()
    f = open('{}_{}_class-fenci.txt'.format(list_number[d],list_number[d+1]), 'w', encoding='utf-8')
    for line in df['new_推文'][list_number[d]:list_number[d+1]]:
        tokens = nltk.word_tokenize(line)
        # 计算关键词
        all_words = tokens
        c = Counter()
        for x in all_words:
            if len(x) > 1 and x != '\r\n':
                c[x] += 1
        # Top20
        output = ""
        # print('\n词频统计结果：')
        for (k, v) in c.most_common(20):
            # print("%s:%d"%(k,v))
            output += k + " "

        f.write(output + "\n")

    else:
        f.close()

    # 文档预料 空格连接
    corpus = []
    # 读取预料 一行预料为一个文档
    for line in open('{}_{}_class-fenci.txt'.format(list_number[d],list_number[d+1]), 'r',encoding='utf-8').readlines():
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
    # weight1 = weight.astype('float16')

    n_clusters = 7
    # 打印特征向量文本内容
    # print('Features length: ' + str(len(word)))
    # #第二步 聚类Kmeans
    # print('Start Kmeans:')

    clf = MiniBatchKMeans(n_clusters=n_clusters)
    pre = clf.fit_predict(weight)
    # print(pre)

    result = pd.concat((df, pd.DataFrame(pre)), axis=1)
    result.rename({0: '聚类结果'}, axis=1, inplace=True)
    result.to_csv('./data/{}_{}_聚类结果.csv'.format(list_number[d],list_number[d+1]),encoding="utf-8-sig")


# #第三步 图形输出 降维
#
# pca = PCA(n_components=n_clusters)  # 输出两维
# newData = pca.fit_transform(weight)  # 载入N维
#
# x = [n[0] for n in newData]
# y = [n[1] for n in newData]
# plt.figure(figsize=(12,9),dpi = 300)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
# plt.rcParams['axes.unicode_minus'] = False
# plt.scatter(x, y, c=pre, s=100)
# plt.title("词性聚类图")
# plt.savefig('词性聚类图.png')
# plt.show()