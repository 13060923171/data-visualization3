import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
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


#LDA建模
def lda():
    #这里的做法都和之前一样，先把数据重新过一遍
    df = pd.read_excel('汽车之家.xlsx')
    df = df.dropna(subset=['全文'], axis=0)
    content = df['全文'].drop_duplicates(keep='first')
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
    f = open('./data/fenci.txt', 'w', encoding='utf-8-sig')
    #接着开始读取数据
    for line in content:
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

    #读取数据
    fr = open('./data/fenci.txt', 'r', encoding='utf-8-sig')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ') if len(word) >= 2]
        train.append(line)
    #构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    # 困惑度模块
    x_data = []
    y_data = []
    z_data = []
    for i in tqdm(range(2, 15)):
        x_data.append(i)
        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, random_state=111, iterations=400)
        # 困惑度计算
        _,perplexity = lda.log_perplexity(corpus)
        y_data.append(perplexity)
        # 一致性计算
        coherencemodel = models.CoherenceModel(model=lda, texts=train, dictionary=dictionary, coherence='c_v')
        z_data.append(coherencemodel.get_coherence())

    # 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 绘制困惑度折线图
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x_data, y_data, marker="o")
    plt.title("perplexity_values")
    plt.xlabel('num topics')
    plt.ylabel('perplexity score')
    #绘制一致性的折线图
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x_data, z_data, marker="o")
    plt.title("coherence_values")
    plt.xlabel("num topics")
    plt.ylabel("coherence score")

    plt.savefig('./data/困惑度和一致性.png')
    plt.show()
    #将上面获取的数据进行保存
    df5 = pd.DataFrame()
    df5['主题数'] = x_data
    df5['困惑度'] = y_data
    df5['一致性'] = z_data
    df5.to_csv('困惑度和一致性.csv',encoding='utf-8-sig',index=False)
    num_topics = input('请输入主题数:')

    #LDA可视化模块
    #构建lda主题参数
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, './data/lda.html')

    #主题判断模块
    list3 = []
    list2 = []
    #这里进行lda主题判断
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
    data['内容'] = content
    data['主题概率'] = list3
    data['主题类型'] = list2

    data.to_csv('./data/new_data.csv',encoding='utf-8-sig',index=False)

    #获取对应主题出现的频次
    new_data = data['主题类型'].value_counts()
    new_data = new_data.sort_index(ascending=True)
    y_data1 = [y for y in new_data.values]

    #主题词模块
    word = lda.print_topics(num_words=20)
    topic = []
    quanzhong = []
    list_gailv = []
    list_gailv1 = []
    list_word = []
    #根据其对应的词，来获取其相应的权重
    for w in word:
        ci = str(w[1])
        c1 = re.compile('\*"(.*?)"')
        c2 = c1.findall(ci)
        list_word.append(c2)
        c3 = '、'.join(c2)

        c4 = re.compile(".*?(\d+).*?")
        c5 = c4.findall(ci)
        for c in c5[::1]:
            if c != "0":
                gailv = str(0) + '.' + str(c)
                list_gailv.append(gailv)
        list_gailv1.append(list_gailv)
        list_gailv = []
        zt = "Topic" + str(w[0])
        topic.append(zt)
        quanzhong.append(c3)

    #把上面权重的词计算好之后，进行保存为csv文件
    df2 = pd.DataFrame()
    for j,k,l in zip(topic,list_gailv1,list_word):
        df2['{}-主题词'.format(j)] = l
        df2['{}-权重'.format(j)] = k
    df2.to_csv('./data/主题词分布表.csv', encoding='utf-8-sig', index=False)

    y_data2 = []
    for y in y_data1:
        number = float(y / sum(y_data1))
        y_data2.append(float('{:0.5}'.format(number)))

    df1 = pd.DataFrame()
    df1['所属主题'] = topic
    df1['文章数量'] = y_data1
    df1['特征词'] = quanzhong
    df1['主题强度'] = y_data2
    df1.to_csv('./data/特征词.csv',encoding='utf-8-sig',index=False)

#绘制主题强度饼图
def plt_pie():
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    df = pd.read_csv('./data/特征词.csv')
    x_data = list(df['所属主题'])
    y_data = list(df['文章数量'])
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('主题强度')
    plt.tight_layout()
    plt.savefig('./data/主题强度.png')


if __name__ == '__main__':
    lda()
    plt_pie()
