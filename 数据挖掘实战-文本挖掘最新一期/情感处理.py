from snownlp import SnowNLP
import pandas as pd
import re
from gensim import corpora, models
import matplotlib.pyplot as plt

def snownlp_fx():
    df = pd.read_csv('数据集.csv')
    content = df['内容'].drop_duplicates(keep='first')
    content = content.dropna(how='any')

    def emotion_scroe(x):
        text = re.sub(r'(?:回复)?(?://)?@[\w\u2E80-\u9FFF]+:?|\[\w+\]', ',', x)
        text = re.sub(r'\n', '', text)
        score = SnowNLP(text)
        fenshu = score.sentiments
        return fenshu

    def type_emotion(x):
        if x < 0.45:
            return 'neg'
        elif 0.45 <= x <= 0.55:
            return 'neu'
        else:
            return 'pos'

    def lda_tfidf():
        fr = open('./data/内容-class-fenci.txt', 'r', encoding='utf-8')
        train = []
        for line in fr.readlines():
            line = [word for word in line.split(' ') if word != '\n']
            train.append(line)

        dictionary = corpora.Dictionary(train)
        dictionary.filter_extremes(no_below=2, no_above=1.0)
        corpus = [dictionary.doc2bow(text) for text in train]
        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=4)

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
        return list3,list2

    list3, list2 = lda_tfidf()
    df1 = pd.DataFrame()
    df1['content'] = content
    df1['emotion_scroe'] = df1['content'].apply(emotion_scroe)
    df1['emotion_type'] = df1['emotion_scroe'].apply(type_emotion)
    df1['主题概率'] = list3
    df1['主题类型'] = list2
    df1.to_csv('new_数据集.csv',encoding='utf-8-sig',index=False)


def topic_pie():
    df = pd.read_csv('new_数据集.csv')
    type1 = df['emotion_type'].value_counts()
    x_data = list(type1.index)
    y_data = list(type1.values)
    plt.figure(figsize=(12, 9), dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('情感分布占比情况')
    plt.savefig('./data/情感分布占比情况.png')

def topic_lda():
    df = pd.read_csv('new_数据集.csv')
    type1 = df['主题类型'].value_counts()
    x_data = list(type1.index)
    y_data = list(type1.values)
    plt.figure(figsize=(12, 9), dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('主题类型占比情况')
    plt.savefig('./data/主题类型占比情况.png')


if __name__ == '__main__':
    snownlp_fx()
    topic_pie()
    topic_lda()
