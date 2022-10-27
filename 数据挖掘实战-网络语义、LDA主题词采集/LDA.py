import pandas as pd
# 数据处理库
import numpy as np
import re
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import Image
import stylecloud
from sklearn.feature_extraction.text import TfidfTransformer
from gensim import corpora, models
import itertools
import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def lda_tfidf():
    # df = pd.read_excel('京东评论.xlsx')
    # df = df.dropna(subset=['评论内容'], axis=0)
    # content = df['评论内容'].drop_duplicates(keep='first')
    # content = content.dropna(how='any')
    #
    #
    # def is_all_chinese(strs):
    #     for _char in strs:
    #         if not '\u4e00' <= _char <= '\u9fa5':
    #             return False
    #     return True
    #
    # stop_words = []
    #
    # with open("stopwords_cn.txt", 'r', encoding='utf-8-sig') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         stop_words.append(line.strip())
    #
    # f = open('class-fenci.txt', 'w', encoding='utf-8-sig')
    # for line in content:
    #     line = line.strip('\n')
    #     # 停用词过滤
    #     line = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", line)
    #     seg_list = jieba.cut(line, cut_all=False)
    #     cut_words = (" ".join(seg_list))
    #
    #     # 计算关键词
    #     all_words = cut_words.split()
    #     c = Counter()
    #     for x in all_words:
    #         if len(x) >= 2 and x != '\r\n' and x != '\n':
    #             if is_all_chinese(x) == True and x not in stop_words:
    #                 c[x] += 1
    #     # Top10
    #     output = ""
    #     # print('\n词频统计结果：')
    #     for (k, v) in c.most_common(10):
    #         # print("%s:%d"%(k,v))
    #         output += k + " "
    #
    #     f.write(output + "\n")
    # else:
    #     f.close()

    fr = open('class-fenci.txt', 'r', encoding='utf-8-sig')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ') if len(word) >= 2]
        train.append(line)

    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    topic_lda = 3

    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_lda)
    word = lda.print_topics(num_words=20)

    # df = pd.DataFrame()
    # topic = []
    # quanzhong = []
    # for w in word:
    #     zt = "Topic" + str(w[0])
    #     topic.append(zt)
    #     quanzhong.append(w[1])
    # df['所属主题'] = topic
    # df['特征词及其权重'] = quanzhong
    # df.to_excel('特征词表.xlsx',encoding='utf-8-sig',index=False)

    topic = []
    quanzhong = []
    for w in word:
        ci = str(w[1])
        c1 = re.compile('\*"(.*?)"')
        c2 = c1.findall(ci)
        c3 = '、'.join(c2)
        zt = "Topic" + str(w[0])
        topic.append(zt)
        quanzhong.append(c3)

    df = pd.DataFrame()
    df['所属主题'] = topic
    df['特征词'] = quanzhong
    df.to_excel('指标.xlsx',encoding='utf-8-sig',index=False)

if __name__ == '__main__':
    lda_tfidf()