import re
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

import random
from PIL import Image
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def word(df,emotion):
    d = {}
    list_text = []
    for t in df['content']:
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
    for key,values in ls[:20]:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_csv('{}-高频词Top20.csv'.format(emotion),encoding='utf-8-sig',index=False)
    x_data.reverse()
    y_data.reverse()

    plt.figure(figsize=(12, 9),dpi=500)
    plt.barh(x_data, y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("{}-高频词Top20".format(emotion))
    plt.xlabel("频次")
    plt.savefig('{}-高频词Top20'.format(emotion))

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl(0, 100%%, %d%%)" % random.randint(20, 50)

    # 读取背景图片
    background_Image = np.array(Image.open('../image.png'))
    text = ' '.join(list_text)
    wc = WordCloud(
        collocations=False,
        font_path='simhei.ttf',  # 中文需设置路径
        margin=1,  # 页面边缘
        mask=background_Image,
        scale=10,
        max_words=100,  # 最多词个数
        random_state=42,
        width=900,
        height=600,
        min_font_size=4,
        max_font_size=80,
        background_color='SlateGray',  # 背景颜色
        color_func=color_func #字体颜色

    )
    # 生成词云
    wc.generate_from_text(text)
    # 存储图像
    wc.to_file("{}-top100-词云图.png".format(emotion))


if __name__ == '__main__':
    list_emotion = ['pos','neg']
    for e in list_emotion:
        df = pd.read_excel('new_bi_lstm.xlsx')
        data = df[df['情感type'] == e]
        word(data,e)









