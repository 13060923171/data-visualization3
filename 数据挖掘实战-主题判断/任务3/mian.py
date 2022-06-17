import pandas as pd
# 数据处理库
import re
from gensim import corpora, models
from snownlp import SnowNLP
import matplotlib.pyplot as plt

def lda_tfidf():
    df = pd.read_csv('公知.csv')
    df = df.dropna(how='any',axis=0)


    fr = open('class-fenci.txt', 'r', encoding='utf-8')
    train = []

    for line in fr.readlines():
        line = [word for word in line.split(' ') if word != '\n']
        train.append(line)

    dictionary = corpora.Dictionary(train)
    dictionary.filter_extremes(no_below=2, no_above=1.0)

    corpus = [dictionary.doc2bow(text) for text in train]

    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=6)

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
        # print(i[bz][0])
        list2.append(i[bz][0])

    df['主题概率'] = list3
    df['主题类型'] = list2
    df.to_csv('new_公知.csv',encoding='utf-8-sig',index=None)


def snownlp_fx():
    df = pd.read_csv('new_公知.csv')

    def emotion_scroe(x):
        text = re.sub(r'(?:回复)?(?://)?@[\w\u2E80-\u9FFF]+:?|\[\w+\]', ',', x)
        text = re.sub(r'\n', '', text)
        score = SnowNLP(text)
        fenshu = score.sentiments
        return fenshu

    df['emotion_scroe'] = df['4'].apply(emotion_scroe)
    df.to_csv('new_公知_情感分析.csv',encoding='utf-8-sig',index=None)


def fx():
    df = pd.read_csv('new_公知_情感分析.csv')

    def emotion_type(x):
        if x > 0.55:
            return 'pos'
        elif x < 0.45:
            return 'neg'
        else:
            return 'neu'

    df['emotion_scroe'] = df['emotion_scroe'].astype(float)
    df['type'] = df['emotion_scroe'].apply(emotion_type)

    def topic_pie(number):
        df1 = df[df['主题类型'] == number]
        type1 = df1['type'].value_counts()
        x_data = list(type1.index)
        y_data = list(type1.values)
        plt.figure(figsize=(12,9),dpi=300)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%',)
        plt.title('主题{}情感分布占比情况'.format(number))
        plt.savefig('主题{}情感分布占比情况.png'.format(number))

    list1 = [0,1,2,3,4,5]
    for l in list1:
        topic_pie(l)

    df2 = df['主题类型'].value_counts()
    x_data = list(df2.index)
    y_data = list(df2.values)
    plt.style.use('ggplot')
    plt.figure(figsize=(9, 6))
    plt.bar(x_data, y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("主题分布情况")
    plt.xlabel("主题")
    plt.ylabel("数量")
    plt.savefig('主题分布情况.png')


if __name__ == '__main__':
    lda_tfidf()
    snownlp_fx()
    fx()