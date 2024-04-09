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

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
import random

def tf_idf(df):
    corpus = []
    for i in df['fenci']:
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
    df2 = df2.sort_values(by=['tfidf'],ascending=False)
    df2.to_csv('./爱奇艺/TF-IDF相关数据.xlsx',encoding='utf-8-sig',index=False)

    df3 = df2.iloc[:30]
    x_data = list(df3['word'])
    y_data = list(df3['tfidf'])
    x_data.reverse()
    y_data.reverse()
    plt.figure(figsize=(12, 9))
    plt.barh(x_data, y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.title("tf-idf 权重最高的top30词汇")
    plt.xlabel("权重")
    plt.savefig('./爱奇艺/tf-idf top30.png')


def word(df):
    d = {}
    list_text = []
    for t in df['fenci']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
                # 添加到列表里面
                list_text.append(i)
                d[i] = d.get(i,0)+1

    ls = list(d.items())
    ls.sort(key=lambda x:x[1],reverse=True)
    x_data = []
    y_data = []
    for key,values in ls[:100]:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_csv('./爱奇艺/高频词Top100.csv',encoding='utf-8-sig',index=False)

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl(0, 100%%, %d%%)" % random.randint(20, 50)

    # 读取背景图片
    background_Image = np.array(Image.open('image.png'))
    # 提取背景图片颜色
    # img_colors = ImageColorGenerator(background_Image)
    text = ' '.join(list_text)
    wc = WordCloud(
        collocations=False,
        font_path='simhei.ttf',  # 中文需设置路径
        margin=1,  # 页面边缘
        mask=background_Image,
        scale=10,
        max_words=100,  # 最多词个数
        random_state=42,
        width=1200,
        height=900,
        min_font_size=4,
        max_font_size=80,
        background_color='SlateGray',  # 背景颜色
        color_func=color_func #字体颜色

    )
    # 生成词云
    wc.generate_from_text(text)
    # 存储图像
    wc.to_file("./爱奇艺/top100-词云图.png")

if __name__ == '__main__':
    df = pd.read_excel('./爱奇艺/新_评论表.xlsx')
    # tf_idf(df)
    word(df)







