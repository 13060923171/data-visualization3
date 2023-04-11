import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from snownlp import SnowNLP
import re
import jieba
import jieba.analyse
from collections import Counter
import itertools
from tqdm import tqdm
import os
import jieba.posseg as posseg

#词频处理
def data_processing():
    df = pd.read_csv('new_data.csv')
    #删除重复行
    data = df.drop_duplicates(keep='first')
    #去掉一些无效词或者标点符号
    def emjio_tihuan(x):
        x1 = str(x)
        x2 = re.sub('(\[.*?\])', "", x1)
        x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
        x4 = re.sub(r'\n', '', x3)
        return x4
    #删除非中文的数据内容
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

        # 读入停用词表
    stop_words = []
    #读取停用词表
    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())
    data['fulltext'] = data['fulltext'].apply(emjio_tihuan)
    data = data.dropna(subset=['fulltext'], axis=0)
    #进行数据分词
    counts = {}
    for d in tqdm(data['fulltext']):
        #分词处理
        res = posseg.cut(d)
        for word, flag in res:
            #条件筛选
            if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                #情感词汇统计
                if flag == 'Ag' or flag == 'a' or flag == 'ad' or flag == 'an':
                #名词统计，景区名词
                # if flag == 'ns':
                    counts[word] = counts.get(word, 0) + 1
    #把字典转换为列表
    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    #读取top100的词汇
    for key, values in ls[:100]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('情感词汇统计.csv', encoding="utf-8-sig")

#情感处理
def snownlp():
    df = pd.read_csv('new_data.csv')
    data = df.drop_duplicates(keep='first')

    def emjio_tihuan(x):
        x1 = str(x)
        x2 = re.sub('(\[.*?\])', "", x1)
        x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
        x4 = re.sub(r'\n', '', x3)
        return x4

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    def get_cut_words(str1):

        # 读入停用词表
        stop_words = []

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        # 分词
        word_num = jieba.lcut(str(str1), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

        word_num_selected = " ".join(word_num_selected)
        #情感打分
        if len(word_num_selected) != 0:
            score = SnowNLP(word_num_selected)
            fenshu = score.sentiments
            return fenshu

        else:
            return np.NAN

    data['fulltext'] = data['fulltext'].apply(emjio_tihuan)
    data = data.dropna(subset=['fulltext'], axis=0)
    data['情感分数'] = data['fulltext'].apply(get_cut_words)
    data = data.dropna(subset=['情感分数'], axis=0)
    #情感级别排序
    def sentiment_type(x):
        x1 = float(x)
        if 0.5 <= x1 < 0.7:
            return '积极-一般'
        elif 0.7 <= x1 < 0.9:
            return '积极-中度'
        elif 0.9 < x1:
            return '积极-高度'
        elif 0.3 <= x1 < 0.5:
            return '消极-一般'
        elif 0.1 <= x1 < 0.3:
            return '消极-中度'
        elif 0.1 > x1:
            return '消极-高度'
    data['情感类型'] = data['情感分数'].apply(sentiment_type)
    new_df = data['情感类型'].value_counts()
    #数据保存
    new_df.to_csv('情感分析.csv',encoding='utf-8-sig')


if __name__ == '__main__':
    data_processing()
    snownlp()


