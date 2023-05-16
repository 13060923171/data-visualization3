import jieba.analyse
from textrank4zh import TextRank4Keyword
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import re
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
import jieba.posseg as posseg
sns.set_style(style="whitegrid")


stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())

def emjio_tihuan(x):
    x1 = str(x)
    x2 = re.sub('(\[.*?\])', "", x1)
    x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
    x4 = re.sub(r'\n', '', x3)
    return x4

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def text_rank(line):
    # 停用词过滤
    line = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", line)
    line = re.sub('(\[.*?\])', "", line)
    line = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', line)
    line = re.sub(r'\n', '', line)
    res = posseg.cut(line)
    str1 = ' '
    for word, flag in res:
        if word not in stop_words and len(word) >= 2:
            if flag == 'Ng' or flag == 'n' or flag == 'nr' or flag == 'ns' or flag == 'nt' or flag == 'nz':
                str1 += word

    tr4w = TextRank4Keyword()
    tr4w.analyze(text=str1, lower=True, window=2)
    word_list = []
    for item in tr4w.get_keywords(30, word_min_len=2):
        word_list.append(item.word)
        # print(item.word, item.weight)
    if len(word_list) != 0:
        return ' '.join(word_list)
    else:
        return np.NAN


def kmeans():
    df = pd.read_excel('data.xlsx')
    df = df.drop_duplicates()
    df['要求全文'] = df['要求全文'].apply(emjio_tihuan)
    df = df.dropna(subset=['要求全文'], axis=0)
    df['text_rank'] = df['要求全文'].apply(text_rank)
    df = df.dropna(subset=['text_rank'], axis=0)

    corpus = []

    for i in df['text_rank']:
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

    silhouette_scores = []
    best_k = None
    # 设置聚类数量K的范围
    range_n_clusters = range(2, 9)
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
    data.to_csv('轮廓系数.csv', encoding="utf-8-sig",index=False)
    # 绘制轮廓系数图
    plt.plot(range_n_clusters, silhouette_scores, 'bo-', alpha=0.8)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for K-means clustering')
    plt.savefig('轮廓系数图.png')
    plt.show()

    n_clusters = best_k
    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))
    # 第二步 聚类Kmeans
    print('Start Kmeans:')

    clf = KMeans(n_clusters=n_clusters, random_state=111)
    pre = clf.fit_predict(weight)

    df['聚类结果'] = list(pre)
    df.to_csv('聚类结果.csv', encoding="utf-8-sig",index=False)


if __name__ == '__main__':
    kmeans()


