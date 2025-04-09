import pandas as pd
import jieba.posseg as pseg
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

import random
from PIL import Image
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def fx_data():
    df = pd.read_excel('new_data.xlsx')
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

    data.to_csv(f'高频词Top300.csv', encoding='utf-8-sig', index=False)


def word_image():
    # 导入停用词列表
    stop_words = []
    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    df = pd.read_csv('高频词Top300.csv')
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
        stopwords=stop_words,
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
    wc.to_file('词云图.png')


def emotion_pie():
    df = pd.read_excel('new_data.xlsx')
    new_df = df['label'].value_counts()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(9, 6), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title(f'情感占比分布')
    plt.tight_layout()
    # 添加图例
    plt.legend(x_data, loc='upper left')
    plt.savefig(f'情感占比分布.png')


if __name__ == '__main__':
    fx_data()
    word_image()
    emotion_pie()