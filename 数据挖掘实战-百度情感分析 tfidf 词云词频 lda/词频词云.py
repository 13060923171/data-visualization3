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



def main3(df,name):
    # data1 = pd.read_excel('negative-TF-IDF相关数据.xlsx')
    # stop_word = data1['word'].tolist()

    d = {}
    list_text = []
    for t in df['fenci']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            if i in stop_word:
                list_text.append(i)
                d[i] = d.get(i, 0) + 1

    ls = list(d.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    for key, values in ls[:100]:
        if key in stop_word:
            x_data.append(key)
            y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_csv(f'{name}-高频词.csv', encoding='utf-8-sig', index=False)

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl({}, 100%, 50%)".format(np.random.randint(0, 300))

    # 读取背景图片
    background_Image = np.array(Image.open('images.png'))
    text = ' '.join(list_text)
    wc = WordCloud(
        collocations=False,  # 禁用词组
        stopwords=stop_word,
        font_path='simhei.ttf',  # 中文字体路径
        margin=3,  # 词云图边缘宽度
        mask=background_Image,  # 背景图形
        scale=3,  # 放大倍数
        max_words=150,  # 最多词个数
        random_state=42,  # 随机状态
        width=800,  # 图片宽度
        height=600,  # 图片高度
        min_font_size=15,  # 最小字体大小
        max_font_size=90,  # 最大字体大小
        background_color='#fdfefe',  # 背景颜色
        color_func=color_func  # 字体颜色函数
    )
    # 生成词云
    wc.generate_from_text(text)
    # 存储图像
    wc.to_file(f'{name}-词云图.png')


if __name__ == '__main__':
    df = pd.read_csv('new_data.csv')
    list_name = ['positive']
    for n in list_name:
        df1 = df[df['label'] == n]
        main3(df1, n)

    main3(df,'整体')
    list_name = ['negative']
    for n in list_name:
        df1 = df[df['label'] == n]
        main3(df1, n)
