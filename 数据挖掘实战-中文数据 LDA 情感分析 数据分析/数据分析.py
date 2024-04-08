import pandas as pd
from snownlp import SnowNLP
import os
import numpy as np

def emotion_type(name):
    df1 = pd.read_csv('./总体-LDA/lda_data.csv')
    df = df1[df1['关键词'] == name]
    def sentiment_score(text):
        s = SnowNLP(text)
        sentiment = s.sentiments
        return round(sentiment,1)

    if not os.path.exists("./{}/".format(name)):
        os.mkdir("./{}/".format(name))
    df['sentiment score'] = df['fenci'].apply(sentiment_score)

    new_df = df['sentiment score'].value_counts()
    new_df = new_df.sort_index()
    new_df.to_csv('./{}/情感分数-柱状图.csv'.format(name),encoding='utf-8-sig')

    def analyze_sentiment(sentiment):
        if sentiment >= 0.5:
            return '积极态度'
        elif sentiment < 0.5:
            return '消极态度'

    df['analyze_sentiment'] = df['sentiment score'].apply(analyze_sentiment)

    def sentiment_word(data,emotion):
        d = {}
        for t in data['fenci']:
            # 把数据分开
            t = str(t).split(" ")
            for i in t:
                # 对文本进行分词和词性标注
                # 添加到列表里面
                d[i] = d.get(i, 0) + 1

        ls = list(d.items())
        ls.sort(key=lambda x: x[1], reverse=True)
        x_data = []
        y_data = []
        for key, values in ls[:50]:
            x_data.append(key)
            y_data.append(values)

        data = pd.DataFrame()
        data['word'] = x_data
        data['counts'] = y_data

        data.to_csv('./{}/{}-高频词Top50-词云图.csv'.format(name,emotion), encoding='utf-8-sig', index=False)

    df1 = df[df['analyze_sentiment'] == '积极态度']
    sentiment_word(df1,'正面')
    df2 = df[df['analyze_sentiment'] == '消极态度']
    sentiment_word(df2, '负面')

    def sentiment_zanbi(data):
        data1 = data
        new_df = data1['analyze_sentiment'].value_counts()
        new_df = new_df.sort_index()
        d = {}
        for key,values in zip(new_df.index,new_df.values):
            d[key] = round(values / sum(list(new_df.values)),2)
        return d

    df_data = df.groupby('主题类型').apply(sentiment_zanbi)

    df_data.to_csv('./{}/情感分数占比-柱状图.csv'.format(name), encoding='utf-8-sig')

def hot_bar():
    df1 = pd.read_csv('./总体-LDA/lda_data.csv')
    df1 = df1.drop_duplicates(subset=['博文内容'])
    new_df = df1.groupby('关键词').agg('sum')
    x_data = [x for x in new_df.index]
    y_list1 = [x for x in new_df['博文点赞']]
    y_list2 = [x for x in new_df['博文评论']]
    y_list3 = [x for x in new_df['博文转发']]
    data = pd.DataFrame()
    data['公园'] = x_data
    data['博文点赞总和'] = y_list1
    data['博文评论总和'] = y_list2
    data['博文转发总和'] = y_list3
    data.to_csv('不同景区的热度对比-原始数据.csv',encoding='utf-8-sig')

def map1(name):
    df1 = pd.read_csv('./总体-LDA/lda_data.csv')
    df = df1[df1['关键词'] == name]
    def process_data(x):
        x1 = str(x).split(" ")
        x1 = x1[0]
        if x1 == '其他' or x1 == '海外':
            return np.NAN
        else:
            return x1
    df['评论ip'] = df['评论ip'].apply(process_data)
    df = df.dropna(subset=['评论ip'],axis=0)
    new_df = df['评论ip'].value_counts()
    new_df.to_csv('./{}/中国地图-原始数据.csv'.format(name), encoding='utf-8-sig')



if __name__ == '__main__':
    df = pd.read_excel('new_data.xlsx')
    new_df = df['关键词'].value_counts()
    x_data = [x for x in new_df.index]
    for x in x_data:
        emotion_type(x)
        map1(x)
    hot_bar()