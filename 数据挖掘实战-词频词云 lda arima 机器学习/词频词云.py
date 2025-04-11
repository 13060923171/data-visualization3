import pandas as pd
import jieba.posseg as pseg
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")
import os
import random
from PIL import Image
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def fx_data(df,name):
    d = {}
    list_text = []
    for t in df['fenci']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            list_text.append(i)
            d[i] = d.get(i, 0) + 1

    ls = list(d.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    for key, values in ls[:300]:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_csv(f'./词频词云/{name}-高频词Top300.csv', encoding='utf-8-sig', index=False)

    word_image(data,name)


def word_image(df,name):
    # 将DataFrame转换为频率字典
    word_freq = pd.Series(df.counts.values, index=df.word).to_dict()

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl({}, 100%, 50%)".format(np.random.randint(0, 300))

    # 读取背景图片
    background_Image = np.array(Image.open('images.png'))
    wc = WordCloud(
        collocations=False,  # 禁用词组
        font_path='simhei.ttf',  # 中文字体路径
        margin=3,  # 词云图边缘宽度
        mask=background_Image,  # 背景图形
        scale=3,  # 放大倍数
        max_words=150,  # 最多词个数
        random_state=42,  # 随机状态
        width=800,  # 提高分辨率
        height=600,
        min_font_size=20,  # 调大最小字体
        max_font_size=100,  # 调大最大字体
        background_color='white',  # 背景颜色
        color_func=color_func  # 字体颜色函数
    )
    # 生成词云
    # 直接从词频生成词云
    wc.generate_from_frequencies(word_freq)
    # 保存高清图片
    wc.to_file(f'./词频词云/{name}-词云图.png')


if __name__ == '__main__':
    if not os.path.exists("./词频词云"):
        os.mkdir("./词频词云")
    df = pd.read_excel('new_data.xlsx')
    # 转换时间列
    df['发布时间'] = pd.to_datetime(df['发布时间'])
    df['quarter'] = df['发布时间'].dt.to_period('Q')
    list_time = ['2024Q2','2024Q3','2024Q4','2025Q1','2025Q2']
    for t in list_time:
        df1 = df[df['quarter'] == t]
        fx_data(df1,t)