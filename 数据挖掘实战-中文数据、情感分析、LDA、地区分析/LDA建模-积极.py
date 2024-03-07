import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import jieba
import jieba.analyse
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import itertools
import pyLDAvis
import pyLDAvis.gensim
from tqdm import tqdm
import os
import jieba
import jieba.posseg as pseg
from gensim.models import LdaModel
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import os

#LDA建模
def lda(df,area,sentiment):
    # 判断是否为中文
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True
    train = []
    stop_words = ["满意","好",'非常','没有','不错','携程','小时','特别','第一天','比较','真的','只能']
    for line in df['评论文本']:
        # 对文本进行分词和词性标注
        words = pseg.cut(line)
        # 保存名词和形容词的列表
        nouns_and_adjs = []
        # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
        for word, flag in words:
            if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                if flag == 'Ag' or flag == 'a' or flag == 'ad' or flag == 'an' or flag == 'n' or flag == 'Ng' or flag == 'nr' or flag == 'ns' or flag == 'nt' or flag == 'nz':
                    # 如果是名词或形容词，就将其保存到列表中
                    nouns_and_adjs.append(word)
        # line = [word.strip(' ') for word in line.split(' ') if len(word) >= 3 and word not in stop_word]
        if len(nouns_and_adjs) != 0:
            train.append(nouns_and_adjs)
        else:
            pass


    #构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    if not os.path.exists("./{}".format(area)):
        os.mkdir("./{}".format(area))
    if not os.path.exists("./{}/{}".format(area,sentiment)):
        os.mkdir("./{}/{}".format(area,sentiment))

    # # 困惑度模块
    # x_data = []
    # y_data = []
    # z_data = []
    # for i in tqdm(range(2, 15)):
    #     x_data.append(i)
    #     lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=i)
    #     # 困惑度计算
    #     perplexity = lda_model.log_perplexity(corpus)
    #     y_data.append(perplexity)
    #     # 一致性计算
    #     coherence_model_lda = CoherenceModel(model=lda_model, texts=train, dictionary=dictionary, coherence='c_v')
    #     coherence = coherence_model_lda.get_coherence()
    #     z_data.append(coherence)
    #
    # # 绘制困惑度和一致性折线图
    # fig = plt.figure(figsize=(15, 5))
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False
    #
    # # 绘制困惑度折线图
    # ax1 = fig.add_subplot(1, 2, 1)
    # plt.plot(x_data, y_data, marker="o")
    # plt.title("perplexity_values")
    # plt.xlabel('num topics')
    # plt.ylabel('perplexity score')
    # #绘制一致性的折线图
    # ax2 = fig.add_subplot(1, 2, 2)
    # plt.plot(x_data, z_data, marker="o")
    # plt.title("coherence_values")
    # plt.xlabel("num topics")
    # plt.ylabel("coherence score")
    #
    # plt.savefig('./{}/{}/困惑度和一致性.png'.format(area,sentiment))
    #
    # #将上面获取的数据进行保存
    # df5 = pd.DataFrame()
    # df5['主题数'] = x_data
    # df5['困惑度'] = y_data
    # df5['一致性'] = z_data
    # df5.to_csv('./{}/{}/困惑度和一致性.csv'.format(area,sentiment),encoding='utf-8-sig',index=False)
    #
    # optimal_z = max(z_data)
    # optimal_z_index = z_data.index(optimal_z)
    # best_topic_number = x_data[optimal_z_index]

    num_topics = 5

    #LDA可视化模块
    #构建lda主题参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, './{}/{}/lda.html'.format(area,sentiment))

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
    # data['内容'] = df['评论分词']
    # data['主题概率'] = list3
    data['主题类型'] = list2

    # data.to_csv('./{}/{}/lda_data.csv'.format(area,sentiment),encoding='utf-8-sig',index=False)

    #获取对应主题出现的频次
    new_data = data['主题类型'].value_counts()
    new_data = new_data.sort_index(ascending=True)
    y_data1 = [y for y in new_data.values]

    #主题词模块
    word = lda.print_topics(num_words=100)
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
    df2.to_csv('./{}/{}/主题词分布表.csv'.format(area,sentiment), encoding='utf-8-sig', index=False)

    # y_data2 = []
    # for y in y_data1:
    #     number = float(y / sum(y_data1))
    #     y_data2.append(float('{:0.5}'.format(number)))
    #
    # df1 = pd.DataFrame()
    # df1['所属主题'] = topic
    # df1['文章数量'] = y_data1
    # df1['特征词'] = quanzhong
    # df1['主题强度'] = y_data2
    # df1.to_csv('./{}/{}/特征词.csv'.format(area,sentiment),encoding='utf-8-sig',index=False)


if __name__ == '__main__':
    list_area = ['东南亚', '欧美', '日韩']
    list_sentiment = ['积极态度']
    df = pd.read_excel('./data/积极.xlsx')
    for l in list_area:
        for s in list_sentiment:
            df1 = df[df['地区分布'] == l]
            df2 = df1[df1['情感分析'] == s]
            lda(df2,l,s)

