import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from snownlp import SnowNLP
import re
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import Image
import stylecloud
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, models
import itertools
import pyLDAvis
import pyLDAvis.gensim
import gensim
from tqdm import tqdm
import os
from gensim.models import LdaModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def data_processing():
    df = pd.read_excel('data.xlsx')
    data = df.drop_duplicates(keep='first')


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

    def get_cut_words(content_series):
        # 读入停用词表
        stop_words = ["请问",'真的',"支持","不用","昨天","只能","感觉","谢谢","安排","你好","告诉","好像","带你去","这是","一点","明天","刚刚","一年","东西","不到","我刚","不上"]

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        # 分词
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    data['评论'] = data['评论'].apply(emjio_tihuan)
    data = data.dropna(subset=['评论'], axis=0)
    text3 = get_cut_words(content_series=data['评论'])
    stylecloud.gen_stylecloud(text=' '.join(text3), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-star',
                              size=500,
                              palette='matplotlib.Inferno_9',
                              output_name='词云图.png')
    Image(filename='词云图.png')

    counts = {}
    for t in text3:
        counts[t] = counts.get(t, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[:100]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('TOP100_高频词.csv', encoding="utf-8-sig")


def snownlp():
    df = pd.read_excel('data.xlsx')
    data = df.drop_duplicates(keep='first')
    data = data.dropna(subset=['评论'], axis=0)

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

    def get_cut_words(str1):

        # 读入停用词表
        stop_words = ["请问", '真的', "支持", "不用", "昨天", "只能", "感觉", "谢谢", "安排", "你好", "告诉", "好像", "带你去", "这是", "一点", "明天",
                      "刚刚", "一年", "东西", "不到", "我刚", "不上"]

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        # 分词
        word_num = jieba.lcut(str(str1), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

        word_num_selected = " ".join(word_num_selected)

        if len(word_num_selected) != 0:
            score = SnowNLP(word_num_selected)
            fenshu = score.sentiments
            return fenshu

        else:
            return np.NAN

    data['评论'] = data['评论'].apply(emjio_tihuan)
    data = data.dropna(subset=['评论'], axis=0)
    data['情感分数'] = data['评论'].apply(get_cut_words)
    data = data.dropna(subset=['情感分数'], axis=0)
    data.to_excel('new_data.xlsx',encoding='utf-8-sig')


def nlp_picture():
    df = pd.read_excel('new_data.xlsx',parse_dates=['评论时间'], index_col="评论时间")
    df1 = df['情感分数'].resample('M').mean()
    df1 = df1.iloc[14:]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(20,9),dpi=300)
    plt.plot(df1,'^--',color='#b82410')
    plt.title('情感月均值趋势')
    plt.xlabel('月份')
    plt.ylabel('均值')
    plt.grid()
    plt.savefig('情感趋势.png')
    plt.show()


#LDA建模
def lda():
    df = pd.read_excel('data.xlsx')
    data = df.drop_duplicates(keep='first')
    data = data.dropna(subset=['评论'], axis=0)

    def emjio_tihuan(x):
        x1 = str(x)
        x2 = re.sub('(\[.*?\])', "", x1)
        x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
        x4 = re.sub(r'\n', '', x3)
        return x4

    data['评论'] = data['评论'].apply(emjio_tihuan)
    data = data.dropna(subset=['评论'], axis=0)

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

        # 读入停用词表
    stop_words = ["请问", '真的', "支持", "不用", "昨天", "只能", "感觉", "谢谢", "安排", "你好", "告诉", "好像", "带你去", "这是", "一点", "明天",
                  "刚刚", "一年", "东西", "不到", "我刚", "不上"]

    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    f = open('class-fenci.txt', 'w', encoding='utf-8-sig')
    for line in data['评论']:
        line = line.strip('\n')
        # 停用词过滤
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
        for (k, v) in c.most_common(30):
            output += k + " "

        f.write(output + "\n")
    else:
        f.close()
    corpus = []
    # 读取预料 一行预料为一个文档
    for line in open('class-fenci.txt', 'r', encoding='utf-8').readlines():
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

    data = {'word': word,
            'tfidf': weight.sum(axis=0).tolist()}
    df2 = pd.DataFrame(data)
    df2['tfidf'] = df2['tfidf'].astype('float64')
    df2 = df2.sort_values(by=['tfidf'], ascending=False)

    x_data = list(df2['word'])[:20]
    y_data = list(df2['tfidf'])[:20]
    x_data.reverse()
    y_data.reverse()
    plt.figure(figsize=(12, 9),dpi=300)
    plt.barh(x_data, y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("TF-IDF TOP20")
    plt.xlabel("数值")
    plt.savefig('TFIDF.png')
    plt.show()
    df2.to_csv('tfidf.csv', encoding='utf-8-sig', index=False)

    fr = open('class-fenci.txt', 'r', encoding='utf-8-sig')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ') if len(word) >= 2]
        train.append(line)

    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    # 构造主题数寻优函数
    def cos(vector1, vector2):  # 余弦相似度函数
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return (None)
        else:
            return (dot_product / ((normA * normB) ** 0.5))

        # 主题数寻优

    def lda_k(x_corpus, x_dict):
        # 初始化平均余弦相似度
        mean_similarity = []
        mean_similarity.append(1)

        # 循环生成主题并计算主题间相似度
        for i in np.arange(2, 11):
            lda = models.LdaModel(x_corpus, num_topics=i, id2word=x_dict)  # LDA模型训练
            for j in np.arange(i):
                term = lda.show_topics(num_words=30)

            # 提取各主题词
            top_word = []
            for k in np.arange(i):

                top_word.append([''.join(re.findall('"(.*)"', i)) \
                                 for i in term[k][1].split('+')])  # 列出所有词

            # 构造词频向量
            word = sum(top_word, [])  # 列出所有的词
            unique_word = set(word)  # 去除重复的词

            # 构造主题词列表，行表示主题号，列表示各主题词
            mat = []
            for j in np.arange(i):
                top_w = top_word[j]
                mat.append(tuple([top_w.count(k) for k in unique_word]))

            p = list(itertools.permutations(list(np.arange(i)), 2))
            l = len(p)
            top_similarity = [0]
            for w in np.arange(l):
                vector1 = mat[p[w][0]]
                vector2 = mat[p[w][1]]
                top_similarity.append(cos(vector1, vector2))

            # 计算平均余弦相似度
            mean_similarity.append(sum(top_similarity) / l)
        return (mean_similarity)

    # 计算主题平均余弦相似度
    word_k = lda_k(corpus, dictionary)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8), dpi=300)
    plt.plot(word_k)
    plt.title('LDA评论主题数寻优')
    plt.xlabel('主题数')
    plt.ylabel('平均余弦相似度')
    plt.savefig('主题数寻优.png')
    plt.show()

    num_topics = input('请输入主题数:')

    #LDA可视化模块
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(data1, 'lda.html')


if __name__ == '__main__':
    data_processing()
    snownlp()
    nlp_picture()
    lda()
