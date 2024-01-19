import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib

from collections import Counter
import itertools
import jieba
import jieba.posseg as pseg

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim.models import LdaModel
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import pyLDAvis.gensim
import pyLDAvis

from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread

matplotlib.use('Agg')

def tf_idf(df,name):
    # corpus = []
    # for i in df['分词']:
    #     corpus.append(i.strip())
    #
    #     # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    # vectorizer = CountVectorizer()
    #
    # # 该类会统计每个词语的tf-idf权值
    # transformer = TfidfTransformer()
    #
    # # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    # tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # # 获取词袋模型中的所有词语
    # word = vectorizer.get_feature_names_out()
    #
    # # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
    # weight = tfidf.toarray()
    #
    # data = {'word': word,
    #         'tfidf': weight.sum(axis=0).tolist()}
    #
    # df2 = pd.DataFrame(data)
    # df2['tfidf'] = df2['tfidf'].astype('float64')
    # df2 = df2.sort_values(by=['tfidf'],ascending=False)
    # df2.to_csv('./需求一/{}_总体_tfidf.csv'.format(name),encoding='utf-8-sig',index=False)

    df2 = pd.read_csv('./需求一/{}_总体_tfidf.csv'.format(name))
    df2['tfidf'] = df2['tfidf'].astype('float64')
    df2 = df2.sort_values(by=['tfidf'],ascending=False)
    df2 = df2.iloc[:100]
    # 导入停用词列表
    stop_words = ['肖战','北京','真的']

    list_word = []
    for i,j in zip(df2['word'],df2['tfidf']):
        if i not in stop_words:
            list_word.append((i, int(j)))
    # c = (
    #     WordCloud()
    #         .add("{}".format(name),list_word, word_size_range=[20, 100], mask_image='中国地图.jpg')
    #         .set_global_opts(title_opts=opts.TitleOpts(title="{}_总体_tf-idf".format(name)))
    #         .render("./需求一/{}_总体_tf-idf.html".format(name))
    # )


    # 将词频数据转换为使用空格隔开的词汇，词频越高的词汇出现次数越多
    text = ' '.join(word for word, freq in list_word for _ in range(freq))

    # 设置中文字体
    font_path = 'C:\Windows\Fonts\simhei.ttf'  # 思源黑体
    # 读取背景图片
    background_Image = np.array(Image.open('中国地图.jpg'))
    # 提取背景图片颜色
    img_colors = ImageColorGenerator(background_Image)

    wc = WordCloud(
        stopwords=STOPWORDS.add("一个"),
        collocations=False,
        font_path=font_path,  # 中文需设置路径
        margin=1,  # 页面边缘
        mask=background_Image,
        scale=10,
        max_words=100,  # 最多词个数
        min_font_size=4,

        random_state=42,
        width=600,
        height=900,
        background_color='SlateGray',  # 背景颜色
        # background_color = '#C3481A', # 背景颜色
        max_font_size=100,

    )
    # 生成词云
    wc.generate_from_text(text)

    # 获取文本词排序，可调整 stopwords
    process_word = WordCloud.process_text(wc, text)
    sort = sorted(process_word.items(), key=lambda e: e[1], reverse=True)

    # 设置为背景色，若不想要背景图片颜色，就注释掉
    wc.recolor(color_func=img_colors)

    # 存储图像
    wc.to_file('./需求一/{}_总体_tf-idf.png'.format(name))

    # # 显示图像
    # plt.imshow(wc, interpolation='bilinear')
    # plt.axis('off')
    # plt.title('{}_总体_tf-idf'.format(name))
    # plt.tight_layout()
    # plt.savefig('./需求一/{}_总体_tf-idf.png'.format(name))
    # plt.show()

    # d = {}
    # for t in df['分词']:
    #     # 把数据分开
    #     t = str(t).split(" ")
    #     for i in t:
    #         # 添加到列表里面
    #         d[i] = d.get(i,0)+1
    #
    # ls = list(d.items())
    # ls.sort(key=lambda x:x[1],reverse=True)
    # x_data = []
    # y_data = []
    # for key,values in ls[:200]:
    #     x_data.append(key)
    #     y_data.append(values)
    #
    # data = pd.DataFrame()
    # data['word'] = x_data
    # data['counts'] = y_data
    #
    # data.to_csv('./需求一/{}_高频词Top200.csv'.format(name),encoding='utf-8-sig',index=False)


#LDA建模
def lda(df,name,emotion):
    new_df = df[df['情感分类'] == emotion]
    train = []
    stop_words = ['肖战', '北京', '真的','肖明','河南','链接','有限公司','哈哈哈','许凯','一种','两个','只能','秦施','杨幂','工作室','二八','视频','小说','微博','孩子','室友','姐姐','刘耀文']
    for line in new_df['分词']:
        line = [word.strip(' ') for word in line.split(' ') if len(word) >= 2 and word not in stop_words]
        train.append(line)

    #构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    #
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
    # plt.savefig('./需求一/{}_{}_困惑度和一致性.png'.format(name,emotion))
    # plt.show()
    # #将上面获取的数据进行保存
    # df5 = pd.DataFrame()
    # df5['主题数'] = x_data
    # df5['困惑度'] = y_data
    # df5['一致性'] = z_data
    # df5.to_csv('./需求一/{}_{}_困惑度和一致性.csv'.format(name,emotion),encoding='utf-8-sig',index=False)
    #
    # optimal_z = max(z_data)
    # optimal_z_index = z_data.index(optimal_z)
    # best_topic_number = x_data[optimal_z_index]

    if name == "前半年" and emotion == "正面情感":
        num_topics = 8
    elif name == "前半年" and emotion == "负面情感":
        num_topics = 2
    elif name == "后半年" and emotion == "正面情感":
        num_topics = 3
    else:
        num_topics = 2

    #LDA可视化模块
    #构建整体参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, './需求一/{}_{}_lda.html'.format(name,emotion))

    #主题判断模块
    list3 = []
    list2 = []
    #这里进行整体判断
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
    df2.to_csv('./需求一/{}_{}_主题词分布表.csv'.format(name, emotion), encoding='utf-8-sig', index=False)

    y_data2 = []
    for y in y_data1:
        number = float(y / sum(y_data1))
        y_data2.append(float('{:0.5}'.format(number)))

    df1 = pd.DataFrame()
    df1['所属主题'] = topic
    df1['文章数量'] = y_data1
    df1['特征词'] = quanzhong
    df1['主题强度'] = y_data2
    df1.to_csv('./需求一/{}_{}_特征词.csv'.format(name, emotion),encoding='utf-8-sig',index=False)

    keyword_list = []
    for d in df1['特征词']:
        d1 = str(d).split('、')
        for l in d1:
            keyword_list.append(l)

    keyword_list1 = list(set(keyword_list))
    with open("./需求一/{}_{}_keyword.txt".format(name, emotion),'w',encoding='utf-8-sig')as f:
        for keyword in keyword_list1:
            f.write(str(keyword) + "\n")


if __name__ == '__main__':
    list_time = ['前半年',"后半年"]
    for t in list_time:
        df = pd.read_csv("{}数据.csv".format(t))
        tf_idf(df,t)

    # list_time = ['前半年',"后半年"]
    # list_emotion = ['正面情感','负面情感']
    #
    # for t in list_time:
    #     for e in tqdm(list_emotion):
    #         df = pd.read_csv("{}数据.csv".format(t))
    #         lda(df,t,e)





