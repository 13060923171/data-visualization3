import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
import stylecloud
from IPython.display import Image
from sklearn.decomposition import LatentDirichletAllocation
import itertools
import matplotlib
import pyLDAvis
import pyLDAvis.gensim
from tqdm import tqdm
import os
from gensim.models import LdaModel
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

df = pd.read_csv('new_data.csv')


def time_process(x):
    x1 = str(x)
    if '分钟' in x1:
        return np.NaN
    else:
        x1 = x1.strip("\n").strip(' ')
        x1 = x1.replace('\n', '').replace('年', '-').replace('月', '-').replace('日', '')
        x1 = x1.split(" ")
        return x1[0]


df['时间'] = df['时间'].apply(time_process)
df.dropna(subset=['时间'], axis=0, inplace=True)
df['时间1'] = pd.to_datetime(df['时间'], format='%Y-%m-%d')
df['发文数量'] = 1


def time_stage():
    new_df = df.groupby('时间1').agg('sum')
    new_df.to_csv('time_data.csv',encoding='utf-8-sig')
    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(new_df,color='#b82410')
    plt.legend()
    plt.title('发帖时间趋势图')
    plt.xlabel('时间')
    plt.ylabel('发帖频次')
    plt.grid()
    plt.savefig('1.png')
    plt.show()


def tf_idf(name,time1):
    df.index = df['时间1']
    df1 = df[time1[0]:time1[1]]

    corpus = []
    # 读取预料 一行预料为一个文档
    for d in df1['分词']:
        corpus.append(d.strip())

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

    data = {'word': word,
            'tfidf': weight.sum(axis=0).tolist()}

    df2 = pd.DataFrame(data)
    df2['tfidf'] = df2['tfidf'].astype('float64')
    df2 = df2.sort_values(by=['tfidf'], ascending=False)
    df3 = df2.iloc[:50]
    if not os.path.exists("./{}".format(name)):
        os.mkdir("./{}".format(name))
    df3.to_csv('./{}/Top50权重词.csv'.format(name), encoding='utf-8-sig', index=False)


def word_cloud(name,time1):
    df.index = df['时间1']
    df1 = df[time1[0]:time1[1]]
    str1 = []
    for d in df1['分词']:
        d = str(d).split(" ")
        for i in d:
            str1.append(i)
    stylecloud.gen_stylecloud(text=' '.join(str1), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-cannabis',
                              size=500,
                              output_name='./{}/词云图.png'.format(name))
    Image(filename='./{}/词云图.png'.format(name))

#LDA建模
def lda(name,time1):
    df.index = df['时间1']
    df1 = df[time1[0]:time1[1]]
    train = []
    for line in df1['分词']:
        line = [word.strip(' ') for word in line.split(' ') if len(word) >= 2]
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

    plt.savefig('./{}/困惑度和一致性.png'.format(name))
    plt.show()
    #将上面获取的数据进行保存
    df5 = pd.DataFrame()
    df5['主题数'] = x_data
    df5['困惑度'] = y_data
    df5['一致性'] = z_data
    df5.to_csv('./{}/困惑度和一致性.csv'.format(name),encoding='utf-8-sig',index=False)
    num_topics = input('请输入主题数:')

    #LDA可视化模块
    #构建lda主题参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, './{}/lda.html'.format(name))


if __name__ == '__main__':
    list_name = ['潜伏期','成长期','成熟期','衰退期']
    list_time = [['2022-05-31','2022-6-14'],['2022-6-15','2022-6-29'],['2022-6-30','2022-7-19'],['2022-7-20','2022-7-31']]
    for n,t in zip(list_name,list_time):
        tf_idf(n,t)
        word_cloud(n,t)
        lda(n,t)