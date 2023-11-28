import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import itertools

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim.models import LdaModel
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import pyLDAvis.gensim
import pyLDAvis

stop_words = []
with open('常用英文停用词(NLP处理英文必备)stopwords.txt','r',encoding='utf-8')as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip().replace("'",""))

def main1():
    #去掉标点符号，以及机械压缩
    def preprocess_word(word):
        # Remove punctuation
        word = word.strip('\'"?!,.():;')
        # Convert more than 2 letter repetitions to 2 letter
        # funnnnny --> funny
        word = re.sub(r'(.)\1+', r'\1\1', word)
        # Remove - & '
        word = re.sub(r'(-|\')', '', word)
        return word

    #再次去掉标点符号
    def gettext(x):
        import string
        punc = string.punctuation
        for ch in punc:
            txt = str(x).replace(ch,"")
        return txt


    #替换表情包
    def handle_emojis(tweet):
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' ', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' ', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', ' ', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' ', tweet)
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' ', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' ', tweet)
        # Angry -- >:(, >:-(, :'(
        tweet = re.sub(r'(>:\(|>:-\(|:\'\()', ' ', tweet)
        # Surprised -- :O, :o, :-O, :-o, :0, 8-0
        tweet = re.sub(r'(:\s?[oO]|:-[oO]|:0|8-0)', ' ', tweet)
        # Confused -- :/, :\, :-/, :-\
        tweet = re.sub(r'(:\\|:/|:-\\\\|:-/)', ' ', tweet)
        # Embarrassed -- :$, :-$
        tweet = re.sub(r'(:\\$|:-\\$)', ' ', tweet)
        # Other emoticons
        # This regex matching method is taken from Twitter NLP utils:
        # https://github.com/aritter/twitter_nlp/blob/master/python/emoticons.py
        emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', tweet)
        for emoticon in emoticons:
            tweet = tweet.replace(emoticon, " ")
        return tweet


    def clean_text(text):
        # Replaces URLs with the word URL
        url_pattern = re.compile(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+')
        text = url_pattern.sub('', text)
        # Replace @username with the word USER_MENTION
        text = re.sub(r'@[\S]+', ' ', text)
        # Replace #hashtag with the word HASHTAG
        text = re.sub(r'#(\S+)', ' ', text)
        # Remove RT (retweet)
        text = re.sub(r'\brt\b', ' ', text)
        # Replace 2+ dots with space
        text = re.sub(r'\.{2,}', ' ', text)
        # Strip space, " and ' from text
        text = text.strip(' "\'')
        # Handle emojis
        text = handle_emojis(text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove numbers
        text = re.sub(r'\d+', ' ', text)
        # Remove punctuation
        text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
        # Lowercase and split into words
        words = text.lower().split()
        words = [w for w in words if w not in stop_words]
        words = [lemmatizer.lemmatize(w) for w in words]
        words = [preprocess_word(w) for w in words]
        if len(words) != 0:
            return ' '.join(words)
        else:
            return np.NAN

    df1 = pd.read_excel('Untitled spreadsheet.xlsx')
    df2 = pd.read_excel('20231019015915065.xlsx')
    df3 = pd.DataFrame()
    df3['公开(公告)号'] = df2['公开(公告)号']
    df3['申请年月'] = df2['申请年月']
    df = pd.merge(df1,df3,on='公开(公告)号')
    df = df.drop_duplicates(subset=['摘要(译)(简体中文)'])
    df = df.drop_duplicates(subset=['标题(译)(简体中文)'])
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['摘要(译)(英文)'] + " " + df['标题(译)(英文)']
    df['summary_and_title'] = df['text'].apply(gettext)
    df['summary_and_title'] = df['summary_and_title'].apply(preprocess_word)
    df['summary_and_title'] = df['summary_and_title'].apply(clean_text)
    new_df = df.dropna(subset=['summary_and_title'],axis=0)

    return new_df

def tf_idf(df):
    corpus = []
    for i in df['summary_and_title']:
        corpus.append(i.strip())

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

    x_data = list(df2['word'])[:30]
    y_data = list(df2['tfidf'])[:30]
    x_data1 = [x for x in x_data]
    x_data1.reverse()
    y_data.reverse()
    plt.figure(figsize=(12, 9), dpi=500)
    plt.barh(x_data1, y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("TF-IDF TOP30")
    plt.xlabel("value")
    plt.savefig('TFIDF.png')
    plt.show()
    df2.to_excel('tfidf.xlsx', index=False)

#LDA建模
def lda(df):
    train = []
    for line in df['summary_and_title']:
        line = [word.strip(' ') for word in line.split(' ') if len(word) >= 2]
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
    # plt.savefig('困惑度和一致性.png')
    # plt.show()
    # #将上面获取的数据进行保存
    # df5 = pd.DataFrame()
    # df5['num topics'] = x_data
    # df5['Perplexity'] = y_data
    # df5['consistency'] = z_data
    # df5.to_csv('困惑度和一致性.csv',encoding='utf-8-sig',index=False)
    #
    # optimal_z = max(z_data)
    # optimal_z_index = z_data.index(optimal_z)
    # best_topic_number = x_data[optimal_z_index]

    num_topics = 3

    #LDA可视化模块
    #构建整体参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, 'lda.html')

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

    def process(x):
        x1 = str(x).split("-")
        return x1[0]

    data = pd.DataFrame()
    data['time_date'] = df['申请年月']
    data['time_date'] = data['time_date'].apply(process)
    data['content'] = df['summary_and_title']
    data['主题概率'] = list3
    data['主题类型'] = list2

    data.to_excel('new_data.xlsx',index=False)

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
    df2.to_csv('主题词分布表.csv', encoding='utf-8-sig', index=False)

    y_data2 = []
    for y in y_data1:
        number = float(y / sum(y_data1))
        y_data2.append(float('{:0.5}'.format(number)))

    df1 = pd.DataFrame()
    df1['所属主题'] = topic
    df1['文章数量'] = y_data1
    df1['特征词'] = quanzhong
    df1['主题强度'] = y_data2
    df1.to_csv('特征词.csv',encoding='utf-8-sig',index=False)


#绘制主题强度饼图
def plt_pie():
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)

    df = pd.read_csv('特征词.csv')
    x_data = [str(x).replace('Topic','') for x in df['所属主题']]
    y_data = list(df['文章数量'])
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('theme strength')
    # 添加图例
    plt.legend(x_data,loc='lower right')
    plt.tight_layout()
    plt.savefig('theme strength.png')


if __name__ == '__main__':
    # df = main1()
    # tf_idf(df)
    # lda(df)
    plt_pie()

