import pandas as pd
# 数据处理库
import numpy as np
import re
import jieba
import jieba.analyse
from collections import Counter
from IPython.display import Image
import stylecloud
from gensim import corpora, models
import itertools
import pyLDAvis
import pyLDAvis.gensim
import os
import matplotlib.pyplot as plt
import paddlehub as hub
import statsmodels.api as sm
import seaborn as sns

#LDA建模
def lda():
    df = pd.read_excel('棱镜.xlsx')
    df = df.dropna(subset=['正文'], axis=0)
    content = df['正文'].drop_duplicates(keep='first')
    content = content.dropna(how='any')

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
    #     # Top30
    #     output = ""
    #     for (k, v) in c.most_common(30):
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
    #困惑度模块
    x_data = []
    y_data = []
    for i in range(2,15):
        x_data.append(i)
        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, random_state=111, iterations=400)
        perplexity = lda.log_perplexity(corpus)
        y_data.append(perplexity)

    data = pd.DataFrame()
    data['主题数'] = x_data
    data['困惑度'] = y_data
    data.to_csv('困惑度.csv',encoding='utf-8-sig',index=False)


    # 绘制困惑度折线图
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.plot(x_data, y_data, marker="o")
    plt.title("主题建模-困惑度")
    plt.xlabel('主题数目')
    plt.ylabel('困惑度大小')
    plt.savefig("主题建模-困惑度.png")
    plt.show()

    #LDA可视化模块
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=11, random_state=111, iterations=400)
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(data1, 'lda.html')

    #主题判断模块
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

    df['主题概率'] = list3
    df['主题类型'] = list2

    df.to_csv('new_困惑度.csv',encoding='utf-8-sig',index=False)
    new_data = df['主题类型'].value_counts()
    new_data = new_data.sort_index(ascending=True)
    y_data1 = [y for y in new_data.values]

    #主题词模块
    word = lda.print_topics(num_words=20)

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

    df1 = pd.DataFrame()
    df1['所属主题'] = topic
    df1['文章数量'] = y_data1
    df1['特征词'] = quanzhong
    df1.to_excel('data.xlsx',encoding='utf-8-sig',index=False)

#情感分析模块
def sentiment():
    df = pd.read_excel('棱镜.xlsx')
    df = df.dropna(subset=['正文'], axis=0)

    # 这里使用了百度开源的成熟NLP模型来预测情感倾向
    senta = hub.Module(name="senta_bilstm")
    texts = df['正文'].tolist()
    input_data = {'text': texts}
    res = senta.sentiment_classify(data=input_data)
    df['情感分值'] = [x['positive_probs'] for x in res]
    df.to_csv('情感数据.csv', encoding='utf-8-sig', index=False)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 6))
    plt.hist(df['情感分值'], bins=np.arange(0, 1, 0.01), facecolor='#E74C3C')
    plt.xlabel('情感数值')
    plt.ylabel('数量')
    plt.title('情感分析')
    plt.savefig('Analysis of Sentiments.jpg')
    plt.show()

#词云图高频词模块，这块是正文分词
def wordclound_zw():
    df = pd.read_excel('棱镜.xlsx')
    df = df.dropna(subset=['正文'], axis=0)
    content = df['正文'].drop_duplicates(keep='first')
    content = content.dropna(how='any')

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

        # 分词
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    text3 = get_cut_words(content_series=content)
    stylecloud.gen_stylecloud(text=' '.join(text3), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-circle',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='正文-词云图.png')
    Image(filename='正文-词云图.png')

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
    df1.to_csv('top200_正文高频词.csv', encoding="utf-8-sig")

#词云图高频词模块，这块是标题分词
def wordclound_bt():
    df = pd.read_excel('棱镜.xlsx')
    df = df.dropna(subset=['新闻标题'], axis=0)
    content = df['新闻标题'].drop_duplicates(keep='first')
    content = content.dropna(how='any')

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

        # 分词
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    text3 = get_cut_words(content_series=content)
    stylecloud.gen_stylecloud(text=' '.join(text3), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-circle',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='标题-词云图.png')
    Image(filename='标题-词云图.png')

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
    df1.to_csv('top200_标题高频词.csv', encoding="utf-8-sig")

#多重线性回归
def topic_pl():
    df = pd.read_csv('new_棱镜.csv')
    y = df['评论数']
    X = df['主题类型']
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())


#相关性
def topic_qg():
    df = pd.read_csv('情感数据.csv')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.regplot(x='情感分值', y='评论数', data=df)
    plt.savefig('相关性.png')
    plt.show()


if __name__ == '__main__':
    lda()
    sentiment()
    wordclound_zw()
    wordclound_bt()
    topic_pl()
    topic_qg()