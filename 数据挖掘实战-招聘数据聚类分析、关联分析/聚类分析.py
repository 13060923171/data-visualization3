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

sns.set_style(style="whitegrid")

df = pd.read_csv('data.csv')
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

#从这里开始，首先创建一个文本
f = open('fenci.txt', 'w', encoding='utf-8-sig')
#接着开始读取数据
for line in df['专业知识']:
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
    df = pd.read_csv('data.csv')
    corpus = []

    # 读取预料 一行预料为一个文档
    for line in open('fenci.txt', 'r', encoding='utf-8-sig').readlines():
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

    d = {}
    plt.figure(figsize=(12, 12),dpi=500)
    for k in range(2, 14):
        est = KMeans(n_clusters=k, random_state=111)
        # 作用到标准化后的数据集上
        y_pred = est.fit_predict(weight)
        # 距离越来越小，效果越来越好
        score = metrics.calinski_harabasz_score(weight, y_pred)
        d.update({k: score})
        print('calinski_harabasz_score with k={0} is {1}'.format(k, score))

    x = []
    y = []
    for k, score in d.items():
        x.append(k)
        y.append(score)

    plt.plot(x, y)
    plt.xlabel('k value')
    plt.ylabel('calinski_harabasz_score')
    plt.savefig('轮廓系数图')

    n_clusters = int(input('请输入需要聚类的参数:'))
    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))
    # 第二步 聚类Kmeans
    print('Start Kmeans:')

    clf = KMeans(n_clusters=n_clusters, random_state=111)
    pre = clf.fit_predict(weight)

    result = pd.concat((df, pd.DataFrame(pre)), axis=1)
    result.rename({0: '聚类结果'}, axis=1, inplace=True)
    result.to_csv('聚类结果.csv', encoding="utf-8-sig",index=False)

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
    plt.savefig('聚类分析图.png')
    plt.show()


if __name__ == '__main__':
    kmeans()