import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import jieba
import jieba.analyse
from collections import Counter
from IPython.display import Image
import stylecloud
import itertools
from tqdm import tqdm
import os
import paddlehub as hub

stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


def snownlp_fx():
    df = pd.read_excel('handle.xlsx')
    data = df.drop_duplicates(keep='first')
    if not os.path.exists("./data"):
        os.mkdir("./data")

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

    # 定义机械压缩函数
    def yasuo(st):
        for i in range(1, int(len(st) / 2) + 1):
            for j in range(len(st)):
                if st[j:j + i] == st[j + i:j + 2 * i]:
                    k = j + i
                    while st[k:k + i] == st[k + i:k + 2 * i] and k < len(st):
                        k = k + i
                    st = st[:j] + st[k:]
        return st

    def get_cut_words(content_series):
        # 读入停用词表
        # 分词
        word_num = jieba.lcut(content_series, cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

        return ' '.join(word_num_selected)

    data['content'] = data['content'].apply(emjio_tihuan)
    data = data.dropna(subset=['content'], axis=0)
    # data['content'] = data['content'].apply(yasuo)
    data['分词'] = data['content'].apply(get_cut_words)
    senta = hub.Module(name="senta_bilstm")
    texts = data['分词'].tolist()
    input_data = {'text': texts}
    res = senta.sentiment_classify(data=input_data)
    data['情感分值'] = [x['positive_probs'] for x in res]
    def main1(x):
        x1 = float(x)
        if x1 <= 0.35:
            return '负面'
        else:
            return '非负'
    data['emotion_type'] = data['情感分值'].apply(main1)
    # data = data.dropna(how='any', axis=0)
    data.to_csv('./data/data_情感分析.csv', encoding='utf-8-sig', index=False)


def wordclound_fx():
    data = pd.read_csv('./data/data_情感分析.csv')
    text1 = [x for x in data['分词']]
    stylecloud.gen_stylecloud(text=' '.join(map(str,text1)), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-globe',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./data/词云图.png')
    Image(filename='./data/词云图.png')

    counts = {}
    for t in text1:
        counts[t] = counts.get(t, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[:100]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./data/高频词TOP100.csv', encoding="utf-8-sig",index=False)


def emotion_score():
    data = pd.read_csv('./data/data_情感分析.csv')
    new_df = data['emotion_type'].value_counts()
    new_df.sort_index(inplace=True)
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(9, 6),dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, shadow=True, startangle=0, autopct='%1.2f%%',
            wedgeprops={'edgecolor': 'black'})
    plt.savefig('./data/情感分布.png')


if __name__ == '__main__':
    snownlp_fx()
    wordclound_fx()
    emotion_score()
