import os
import pandas as pd
import numpy as np
from IPython.display import Image
import stylecloud
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="whitegrid")


#中文判断函数#
def wordclound_fx(name=None):
    df = pd.read_csv('聚类结果.csv')
    df = df[df['聚类结果'] == name]
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    def get_cut_words(content_series):
        # 读入停用词表
        stop_words = []

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        # 分词
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    text3 = get_cut_words(content_series=df['专业知识'])
    stylecloud.gen_stylecloud(text=' '.join(text3), max_words=150,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-circle',
                              # icon_name='fas fa-star',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./data/聚类{}-词云图.png'.format(name))
    Image(filename='./data/聚类{}-词云图.png'.format(name))

    counts = {}
    for t in text3:
        counts[t] = counts.get(t, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[:150]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./data/聚类{}TOP150_高频词.csv'.format(name), encoding="utf-8-sig")


def academic_qualifications(name):
    df = pd.read_csv('聚类结果.csv')
    df = df[df['聚类结果'] == name]
    def main1(x):
        x1 = str(x)
        x1 = x1.replace('统招本科','本科').replace('中专/中技','中专').replace('nan','学历不限').replace('初中及以下','学历不限').replace('MBA/EMBA','博士').replace('EMBA','博士')
        x1 = x1.replace('中专','大专以下').replace('高中','大专以下')
        if '及以上' in x1:
            x2 = x1.split('及以上')
            return x2[0]
        else:
            return x1
    df['学历'] = df['学历'].apply(main1)
    new_df1 = df['学历'].value_counts()
    x_data = [x for x in new_df1.index]
    y_data = [y for y in new_df1.values]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6),dpi=500)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%',
            wedgeprops={'edgecolor': 'black'})
    plt.title('聚类{}-学历占比'.format(name))
    plt.tight_layout()
    plt.savefig('./data/聚类{}-学历占比.png'.format(name))

    bfb = []
    for x in y_data:
        x1 = x / sum(y_data)
        x1 = round(x1,3)
        bfb.append(x1)

    data = pd.DataFrame()
    data['word'] = x_data
    data['count'] = y_data
    data['占比'] = bfb
    data.to_csv('./data/聚类{}-学历占比.csv'.format(name),encoding='utf-8-sig')


def experience(name):
    df = pd.read_csv('聚类结果.csv')
    df = df[df['聚类结果'] == name]
    def main1(x):
        x1 = str(x)
        x1 = x1.replace('1年以下','经验不限').replace('一年以下','经验不限')
        x1 = x1.replace('nan','经验不限').replace('无经验','经验不限')
        x1 =  x1.replace('经验不限', '不限')
        return x1
    df['经验'] = df['经验'].apply(main1)
    new_df1 = df['经验'].value_counts()
    x_data = [x for x in new_df1.index]
    y_data = [y for y in new_df1.values]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6),dpi=500)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%',
            wedgeprops={'edgecolor': 'black'})
    plt.title('聚类{}-经验占比'.format(name))
    plt.tight_layout()
    plt.savefig('./data/聚类{}-经验占比.png'.format(name))

    bfb = []
    for x in y_data:
        x1 = x / sum(y_data)
        x1 = round(x1, 3)
        bfb.append(x1)

    data = pd.DataFrame()
    data['word'] = x_data
    data['count'] = y_data
    data['占比'] = bfb
    data.to_csv('./data/聚类{}-经验占比.csv'.format(name), encoding='utf-8-sig')

def major(name):
    df = pd.read_csv('聚类结果.csv')
    df = df[df['聚类结果'] == name]

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    def get_cut_words(content_series):
        # 读入停用词表
        stop_words = []

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())
        stop_words.remove('不限')
        # 分词
        jieba.load_userdict('dict.txt')
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)
        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    text3 = get_cut_words(content_series=df['专业'])

    counts = {}
    for t in text3:
        t = t.split('类')[0]
        t = t.replace('金融类','金融').replace('财务','会计').replace('财会','会计').replace('财经类','会计')
        counts[t] = counts.get(t, 0) + 1

    data = pd.read_excel('./data/专业词汇.xlsx')
    z_data = list(data['word'])
    x_data = []
    y_data = []
    for z in z_data:
        try:
            values = counts[z]
            x_data.append(z)
            y_data.append(values)
        except:
            pass
    x_data1 = []
    y_data1 = []
    sum1 = 0
    for j,k in zip(x_data,y_data):
        if k < 200:
            sum1 += k
        else:
            x_data1.append(j)
            y_data1.append(k)
    x_data1.append('其他')
    y_data1.append(sum1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6), dpi=500)
    plt.pie(y_data1, labels=x_data1, startangle=0, autopct='%1.2f%%',
            wedgeprops={'edgecolor': 'black'})
    plt.title('聚类{}-专业占比'.format(name))
    plt.tight_layout()
    plt.savefig('./data/聚类{}-专业占比.png'.format(name))



    bfb = []
    for x in y_data1:
        x1 = x / sum(y_data1)
        x1 = round(x1, 3)
        bfb.append(x1)

    data = pd.DataFrame()
    data['word'] = x_data1
    data['count'] = y_data1
    data['占比'] = bfb
    data.to_csv('./data/聚类{}-专业占比.csv'.format(name), encoding='utf-8-sig')


def professional_knowledge(name):
    df = pd.read_csv('聚类结果.csv')
    df = df[df['聚类结果'] == name]
    # def is_all_chinese(strs):
    #     for _char in strs:
    #         if not '\u4e00' <= _char <= '\u9fa5':
    #             return False
    #     return True
    #
    # def get_cut_words(content_series):
    #     # 读入停用词表
    #     stop_words = []
    #
    #     with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             stop_words.append(line.strip())
    #     word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)
    #     # 条件筛选
    #     word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 4 and is_all_chinese(i) == True]
    #     return word_num_selected
    #
    # text3 = get_cut_words(content_series=df['专业知识'])
    #
    # counts = {}
    # for t in text3:
    #     counts[t] = counts.get(t, 0) + 1
    #
    # ls = list(counts.items())
    # ls.sort(key=lambda x: x[1], reverse=True)
    # x_data = []
    # y_data = []
    #
    # for key, values in ls[:100]:
    #     x_data.append(key)
    #     y_data.append(values)
    #
    # df1 = pd.DataFrame()
    # df1['word'] = x_data
    # df1['counts'] = y_data
    # df1.to_csv('./data/总top100_高频词.csv', encoding="utf-8-sig")

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    def get_cut_words(content_series):
        # 读入停用词表
        stop_words = []

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)
        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 4 and is_all_chinese(i) == True]
        return word_num_selected

    text3 = get_cut_words(content_series=df['专业知识'])

    counts = {}
    for t in text3:
        counts[t] = counts.get(t, 0) + 1

    data = pd.read_csv('./data/专业词汇2.csv')
    z_data = list(data['word'])
    x_data = []
    y_data = []
    for z in z_data:
        try:
            values = counts[z]
            x_data.append(z)
            y_data.append(values)
        except:
            pass

    d = {}
    for x,y in zip(x_data,y_data):
        d[x] = y
    ls = list(d.items())
    ls.sort(key=lambda x:x[1],reverse=True)
    ls = ls[0:10]

    x_data1 = []
    y_data1 = []
    for key,values in ls:
        x_data1.append(key)
        y_data1.append(values)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6), dpi=500)
    plt.pie(y_data1, labels=x_data1, startangle=0, autopct='%1.2f%%',
            wedgeprops={'edgecolor': 'black'})
    plt.title('聚类{}-专业知识TOP10占比'.format(name))
    plt.tight_layout()
    plt.savefig('./data/聚类{}-专业知识TOP10占比.png'.format(name))

    bfb = []
    for x in y_data1:
        x1 = x / sum(y_data1)
        x1 = round(x1, 3)
        bfb.append(x1)

    data = pd.DataFrame()
    data['word'] = x_data1
    data['count'] = y_data1
    data['占比'] = bfb
    data.to_csv('./data/聚类{}-专业知识TOP10占比.csv'.format(name), encoding='utf-8-sig')

def corr1():
    data = pd.read_csv('./data/专业词汇2.csv')
    data = data.sort_values(by=['counts'], ascending=False)
    x_data = [x for x in data['word']]
    y_data = [y for y in data['counts']]
    x_data = x_data[:10]
    y_data = y_data[:10]

    df = pd.read_csv('聚类结果.csv')
    count = len(df)
    new_df = df['聚类结果'].value_counts()
    new_df = new_df.sort_index()
    list1 = [x for x in new_df.index]

    def main1(name):
        df = pd.read_csv('聚类结果.csv')
        df = df[df['聚类结果'] == name]
        count = len(df)
        def is_all_chinese(strs):
            for _char in strs:
                if not '\u4e00' <= _char <= '\u9fa5':
                    return False
            return True

        def get_cut_words(content_series):
            # 读入停用词表
            stop_words = []

            with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    stop_words.append(line.strip())
            word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)
            # 条件筛选
            word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 4 and is_all_chinese(i) == True]
            return word_num_selected

        text3 = get_cut_words(content_series=df['专业知识'])

        counts = {}
        for t in text3:
            counts[t] = counts.get(t, 0) + 1

        x_data1 = []
        y_data1 = []

        for x in x_data:
            try:
                values = counts[x]
                x_data1.append(x)
                y_data1.append(values)
            except:
                x_data1.append(x)
                y_data1.append(0)

        return x_data1,y_data1,count

    x_data2 = []
    y_data2 = []
    c_counts = []
    for l in list1:
        x_data1,y_data1,counts = main1(l)
        x_data2.append(x_data1)
        y_data2.append(y_data1)
        c_counts.append(counts)


    y_data3 = []
    for x,y in zip(y_data,y_data2[0]):
        x2 = float(int(y) / int(c_counts[0])) / float(int(x) / int(count))
        x2 = round(x2,3)
        y_data3.append(x2)

    y_data4 = []
    for x, y in zip(y_data, y_data2[1]):
        x2 = float(int(y) / int(c_counts[0])) / float(int(x) / int(count))
        x2 = round(x2, 3)
        y_data4.append(x2)

    y_data5 = []
    for x, y in zip(y_data, y_data2[2]):
        x2 = float(int(y) / int(c_counts[0])) / float(int(x) / int(count))
        x2 = round(x2, 3)
        y_data5.append(x2)

    data2 = pd.DataFrame()
    data2['聚类0'] = y_data3
    data2['聚类1'] = y_data4
    data2['聚类2'] = y_data5
    data2.index = x_data2[0]
    data2 = data2.T
    data2.to_csv('./data/相关性.csv',encoding='utf-8-sig')



if __name__ == '__main__':
    df = pd.read_csv('聚类结果.csv')
    new_df = df['聚类结果'].value_counts()
    list1 = [x for x in new_df.index]
    for l in list1:
        wordclound_fx(l)
        academic_qualifications(l)
        experience(l)
        major(l)
        professional_knowledge(l)
    corr1()