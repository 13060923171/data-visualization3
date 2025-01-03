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

def snownlp_fx():
    df = pd.read_excel('所有评论.xlsx')
    df = df.dropna(subset=['text'],axis=0)
    df = df[df['keyword'] == '沈阳']
    content = df['text'].drop_duplicates(keep='first')
    content = content.dropna(how='any')

    def emotion_scroe(x):
        text = re.sub(r'(?:回复)?(?://)?@[\w\u2E80-\u9FFF]+:?|\[\w+\]', ',', x)
        text = re.sub(r'\n', '', text)
        score = SnowNLP(text)
        fenshu = score.sentiments
        return fenshu

    df1 = pd.DataFrame()
    df1['content'] = content
    df1['emotion_scroe'] = df1['content'].apply(emotion_scroe)
    if not os.path.exists("./沈阳-data"):
        os.mkdir("./沈阳-data")
    df1.to_csv('./沈阳-data/沈阳-snownlp情感分析.csv', encoding='utf-8-sig', index=False)


def wordclound_fx():
    df = pd.read_csv('./沈阳-data/沈阳-snownlp情感分析.csv')
    df1 = df[df['emotion_scroe'] >= 0.5]
    df2 = df[df['emotion_scroe'] < 0.5]

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    def get_cut_words(content_series):
        # 读入停用词表
        stop_words = []

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        # new_stop_words = []
        # with open("停用词1.txt", 'r', encoding='utf-8') as f:
        #     lines = f.readlines()
        # lines = lines[0].split('、')
        # for line in lines:
        #     new_stop_words.append(line.strip())
        # stop_words.extend(new_stop_words)

        # 分词
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    text1 = get_cut_words(content_series=df1['content'])
    stylecloud.gen_stylecloud(text=' '.join(text1), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-circle',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./沈阳-data/正向词云图.png')
    Image(filename='./沈阳-data/正向词云图.png')

    text2 = get_cut_words(content_series=df2['content'])
    stylecloud.gen_stylecloud(text=' '.join(text2), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-anchor',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./沈阳-data/负向词云图.png')
    Image(filename='./沈阳-data/负向词云图.png')

    text3 = get_cut_words(content_series=df['content'])
    stylecloud.gen_stylecloud(text=' '.join(text3), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-atom',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./沈阳-data/总体词云图.png')
    Image(filename='./沈阳-data/总体词云图.png')

    counts = {}
    for t in text3:
        counts[t] = counts.get(t, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[:200]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./沈阳-data/高频词.csv', encoding="utf-8-sig")

    counts1 = {}
    for t in text1:
        counts1[t] = counts1.get(t, 0) + 1

    ls = list(counts1.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[:200]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./沈阳-data/正向-高频词.csv', encoding="utf-8-sig")

    counts2 = {}
    for t in text2:
        counts2[t] = counts2.get(t, 0) + 1

    ls = list(counts2.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[:200]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./沈阳-data/负向-高频词.csv', encoding="utf-8-sig")


def lda_tfidf():
    df = pd.read_csv('./沈阳-data/沈阳-snownlp情感分析.csv')
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    stop_words = []

    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    # new_stop_words = []
    # with open("停用词2.txt", 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    # lines = lines[0].split('、')
    # for line in lines:
    #     new_stop_words.append(line.strip())
    # stop_words.extend(new_stop_words)

    # jieba.analyse.set_stop_words(stop_words)
    f = open('./沈阳-data/沈阳-class-fenci.txt', 'w', encoding='utf-8')
    for line in df['content']:
        line = line.strip('\n')
        # 停用词过滤
        line = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", line)
        seg_list = jieba.cut(line, cut_all=False)
        cut_words = (" ".join(seg_list))

        # 计算关键词
        all_words = cut_words.split()
        c = Counter()
        for x in all_words:
            if len(x) > 1 and x != '\r\n':
                if is_all_chinese(x) == True and x not in stop_words:
                    c[x] += 1
        # Top50
        output = ""
        # print('\n词频统计结果：')
        for (k, v) in c.most_common():
            # print("%s:%d"%(k,v))
            output += k + " "

        f.write(output + "\n")
    else:
        f.close()

    corpus = []

    # 读取预料 一行预料为一个文档
    for line in open('./沈阳-data/沈阳-class-fenci.txt', 'r', encoding='utf-8').readlines():
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
    df2.to_csv('./沈阳-data/沈阳-tfidf.csv', encoding='utf-8-sig', index=False)

    fr = open('./沈阳-data/沈阳-class-fenci.txt', 'r', encoding='utf-8')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ')]
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
                term = lda.show_topics(num_words=50)

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
    plt.savefig('./沈阳-data/LDA评论主题数寻优.png')
    plt.show()

    topic_lda = word_k.index(min(word_k)) + 1

    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_lda)

    data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(data, './沈阳-data/沈阳-lda.html')


if __name__ == '__main__':
    snownlp_fx()
    wordclound_fx()
    lda_tfidf()