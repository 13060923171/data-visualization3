import re
import pyLDAvis
import pyLDAvis.gensim
from tqdm import tqdm

import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel

from tqdm import tqdm
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from maha.cleaners.functions import numbers_to_text
from maha.cleaners.functions import normalize, keep, remove

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('arabic')

# 定义预处理函数
def preprocess_text(text, stop_words):
    # 使用maha库进行预处理
    text = remove(text, punctuations=True)  # 去除多余的空格
    text = keep(text, arabic_letters=True, harakat=True)  # 保持仅阿拉伯语字符
    text = numbers_to_text(text, accusative=True)  # 清理文本，去除URL、标点符号、数字等
    text = normalize(str(text), alef=False, all=True)  # 艾利夫修正
    text = [word for word in word_tokenize(text) if word not in stop_words]  # 去除停用词
    stemmer = ISRIStemmer()
    text = [stemmer.stem(word) for word in text]  # 词干提取
    text1 = ' '.join(text)
    return text1

# 示例停用词列表
stop_words1 = nltk.corpus.stopwords.words('arabic').copy()

stop_words2 = []
with open("list.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words2.append(line.strip())

stop_words = stop_words1 + stop_words2

df = pd.read_excel('data.xlsx')
df = df.drop_duplicates(subset=['content'])
df = df.dropna(subset=['content'], axis=0)

# 预处理推文
processed_tweets = [preprocess_text(tweet, stop_words) for tweet in tqdm(df['content'].tolist())]
df['fenci'] = processed_tweets
df.to_excel('new_data.xlsx', index=False)


def lda(processed_tweets):
    train = []
    for line in processed_tweets:
        line = [word.strip(' ') for word in line.split(' ') if word not in stop_words]
        train.append(line)

    #构建为字典的格式
    dictionary = corpora.Dictionary(train)
    # 过滤掉出现次数太少的词
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    # 创建文档-词频矩阵
    corpus = [dictionary.doc2bow(text) for text in train]
    #
    # # 困惑度模块
    # x_data = []
    # y_data = []
    # z_data = []
    # for i in tqdm(range(2, 16)):
    #     print(i)
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
    # plt.savefig('困惑度和一致性.png')
    #
    # #将上面获取的数据进行保存
    # df5 = pd.DataFrame()
    # df5['主题数'] = x_data
    # df5['困惑度'] = y_data
    # df5['一致性'] = z_data
    # df5.to_csv('困惑度和一致性.csv',encoding='utf-8-sig',index=False)
    #
    # optimal_z = max(z_data)
    # optimal_z_index = z_data.index(optimal_z)
    # best_topic_number = x_data[optimal_z_index]

    num_topics = 4
    #LDA可视化模块
    #构建lda主题参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, 'lda.html')

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

    df.to_excel('lda_new_data.xlsx',index=False)

if __name__ == '__main__':
    lda(processed_tweets)