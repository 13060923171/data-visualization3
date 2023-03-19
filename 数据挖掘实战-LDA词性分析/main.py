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
import jieba.posseg as posseg

def data_processing():
    df = pd.read_excel('故宫博物院.xlsx')
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

        # 读入停用词表
    stop_words = []

    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    def get_cut_words(content_series):

        # 分词
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    data['评论内容'] = data['评论内容'].apply(emjio_tihuan)
    data = data.dropna(subset=['评论内容'], axis=0)
    # text3 = get_cut_words(content_series=data['评论内容'])
    # stylecloud.gen_stylecloud(text=' '.join(text3), max_words=100,
    #                           collocations=False,
    #                           font_path='simhei.ttf',
    #                           icon_name='fas fa-star',
    #                           size=500,
    #                           palette='matplotlib.Inferno_9',
    #                           output_name='词云图.png')
    # Image(filename='词云图.png')
    counts = {}
    for d in data['评论内容']:
        res = posseg.cut(d)
        for word, flag in res:
            if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                # if flag == 'Ag' or flag == 'a' or flag == 'ad' or flag == 'an' or flag == 'Ng' or flag == 'n' or flag == 'nr' or flag == 'ns' or flag == 'nt' or flag == 'nz':
                if flag == 'Ng' or flag == 'n' or flag == 'nr' or flag == 'ns' or flag == 'nt' or flag == 'nz':
                    counts[word] = counts.get(word, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('名词.csv', encoding="utf-8-sig")


def snownlp():
    df = pd.read_excel('故宫博物院.xlsx')
    data = df.drop_duplicates(keep='first')
    data = data.dropna(subset=['评论内容'], axis=0)

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
        stop_words = []

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        word_num_selected = []
        res = posseg.cut(str1)
        for word, flag in res:
            if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                if flag == 'Ag' or flag == 'a' or flag == 'ad' or flag == 'an' or flag == 'Ng' or flag == 'n' or flag == 'nr' or flag == 'ns' or flag == 'nt' or flag == 'nz':
                    word_num_selected.append(word)
        # # 分词
        # word_num = jieba.lcut(str(str1), cut_all=False)
        #
        # # 条件筛选
        # word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

        word_num_selected = " ".join(word_num_selected)

        if len(word_num_selected) != 0:
            score = SnowNLP(word_num_selected)
            fenshu = score.sentiments
            return fenshu

        else:
            return np.NAN

    data['评论内容'] = data['评论内容'].apply(emjio_tihuan)
    data = data.dropna(subset=['评论内容'], axis=0)
    data['情感分数'] = data['评论内容'].apply(get_cut_words)
    data = data.dropna(subset=['情感分数'], axis=0)

    def sentiment_type(x):
        x1 = float(x)
        if x1 > 0.6:
            return 'pos'
        elif 0.4 <= x1 <= 0.6:
            return 'neu'
        else:
            return 'neg'

    data['情感类型'] = data['情感分数'].apply(sentiment_type)
    data.to_excel('new_data.xlsx',encoding='utf-8-sig',index=False)


def nlp_picture():
    df = pd.read_excel('new_data.xlsx',parse_dates=['评论时间'], index_col="评论时间")
    new_df = df['情感类型'].value_counts()
    print('评论总数:',len(df['情感类型']))
    print('正面评论数量:{},正面评论占比:{}'.format(new_df.values[0],new_df.values[0] / len(df['情感类型'])))
    print('中立评论数量:{},中立评论占比:{}'.format(new_df.values[2], new_df.values[2] / len(df['情感类型'])))
    print('正面评论数量:{},正面评论占比:{}'.format(new_df.values[1], new_df.values[1] / len(df['情感类型'])))

    df1 = df['情感分数'].resample('M').mean()
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
    df = pd.read_excel('new_data.xlsx')
    new_df = df[df['情感类型'] == 'pos']
    data = new_df.drop_duplicates(keep='first')
    data = data.dropna(subset=['评论内容'], axis=0)

    def emjio_tihuan(x):
        x1 = str(x)
        x2 = re.sub('(\[.*?\])', "", x1)
        x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
        x4 = re.sub(r'\n', '', x3)
        return x4

    data['评论内容'] = data['评论内容'].apply(emjio_tihuan)
    data = data.dropna(subset=['评论内容'], axis=0)

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

        # 读入停用词表
    stop_words = []

    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())


    list1 = []
    list2 = []
    dit = {}
    for line in data['评论内容']:
        line = line.strip('\n')
        # 停用词过滤
        res = posseg.cut(line)
        for word, flag in res:
            if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                if flag == 'Ag' or flag == 'a' or flag == 'ad' or flag == 'an':
                    dit[word] = dit.get(word,0)+1
                    list1.append(word)
        list2.append(list1)
        list1 = []

    ls = list(dit.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('正面形容词.csv', encoding="utf-8-sig")

    with open('class-fenci-neg.txt', 'w', encoding='utf-8-sig') as f:
        for l in list2:
            list_seg = ' '.join(l)
            f.write(list_seg + "\n")


    fr = open('class-fenci-neg.txt', 'r', encoding='utf-8-sig')
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
    plt.savefig('正面-主题数寻优.png')
    plt.show()

    num_topics = input('请输入主题数:')

    #LDA可视化模块
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(data1, '正面-lda.html')

    # 主题判断模块
    list3 = []
    list2 = []
    for i in lda.get_document_topics(corpus)[:]:
        listj = []
        list1 = []
        for j in i:
            list1.append(j)
            listj.append(j[1])
        list3.append(list1)
        bz = listj.index(max(listj))
        list2.append(i[bz][0])

    data = pd.DataFrame()
    data['主题概率'] = list3
    data['主题类型'] = list2

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)

    new_df = data['主题类型'].value_counts()
    x_data = list(new_df.index)
    y_data = list(new_df.values)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('正面-主题占比')
    plt.tight_layout()
    plt.savefig('正面-主题占比.png')




if __name__ == '__main__':
    # data_processing()
    # snownlp()
    # nlp_picture()
    lda()
