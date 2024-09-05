import re
import os
import numpy as np
import pandas as pd

# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

import pyLDAvis
import pyLDAvis.gensim
from tqdm import tqdm

import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('wordnet')

#LDA建模
def lda(df):
    train = []
    # 加载NLTK的停用词列表
    nltk_stop_words = set(stopwords.words('english'))
    stop_words = []
    with open('stopwords_en.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip().replace("'", ""))

    lemmatizer = WordNetLemmatizer()

    stop_words = set(stop_words)
    for line in df['分词']:
        # 预处理：移除空格，替换'-'，转换为小写

        line = [word.strip(' ').replace('-', '').lower() for word in line.split(' ')
                if len(word.strip().replace('-', '').lower()) > 3 and word not in stop_words and word not in nltk_stop_words]
        # 应用lemmatizer进行词形还原
        line = [lemmatizer.lemmatize(word) for word in line]
        train.append(line)

    #构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    if not os.path.exists("./LDA-EN/"):
        os.mkdir("./LDA-EN/")

    # 困惑度模块
    x_data = []
    y_data = []
    z_data = []
    for i in tqdm(range(2, 16)):
        x_data.append(i)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=i)
        # 困惑度计算
        perplexity = lda_model.log_perplexity(corpus)
        y_data.append(perplexity)
        # 一致性计算
        coherence_model_lda = CoherenceModel(model=lda_model, texts=train, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model_lda.get_coherence()
        z_data.append(coherence)

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

    plt.savefig('./LDA-EN/困惑度和一致性.png')

    #将上面获取的数据进行保存
    df5 = pd.DataFrame()
    df5['主题数'] = x_data
    df5['困惑度'] = y_data
    df5['一致性'] = z_data
    df5.to_csv('./LDA-EN/困惑度和一致性.csv',encoding='utf-8-sig',index=False)

    optimal_z = max(z_data)
    optimal_z_index = z_data.index(optimal_z)
    best_topic_number = x_data[optimal_z_index]

    num_topics = best_topic_number
    # num_topics = 8
    #LDA可视化模块
    #构建lda主题参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, './LDA-EN/lda.html')

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

    df['主题概率'] = list3
    df['主题类型'] = list2

    df.to_csv('./LDA-EN/lda_data.csv',encoding='utf-8-sig',index=False)

    #获取对应主题出现的频次
    new_data = df['主题类型'].value_counts()
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
    df2.to_csv('./LDA-EN/主题词分布表.csv', encoding='utf-8-sig', index=False)

    y_data2 = []
    for y in y_data1:
        number = float(y / sum(y_data1))
        y_data2.append(float('{:0.5}'.format(number)))

    df1 = pd.DataFrame()
    df1['所属主题'] = topic
    df1['文章数量'] = y_data1
    df1['特征词'] = quanzhong
    df1['主题强度'] = y_data2
    df1.to_csv('./LDA-EN/特征词.csv',encoding='utf-8-sig',index=False)


# 去掉标点符号，以及机械压缩
def preprocess_word(word):
    word = word.strip('\'"?!,.():;')
    word = re.sub(r'(.)\1+', r'\1\1', word)
    word = re.sub(r'(-|\')', '', word)
    return word


if __name__ == '__main__':
    df = pd.read_excel('en_comment.xlsx')
    df['分词'] = df['分词'].apply(preprocess_word)
    lda(df)