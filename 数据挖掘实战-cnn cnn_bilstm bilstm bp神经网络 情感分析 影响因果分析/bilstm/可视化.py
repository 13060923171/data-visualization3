import re
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

# import random
# from PIL import Image
# from matplotlib.pyplot import imread
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import pyecharts.options as opts
from pyecharts.charts import Bar
from pyecharts.charts import WordCloud

# 导入停用词列表
stop_words = ['满意','不满意']
with open("../stopwords_cn.txt", 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())

def word(df,emotion):
    d = {}
    list_text = []
    for t in df['content']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            if i not in stop_words:
                # 添加到列表里面
                list_text.append(i)
                d[i] = d.get(i,0)+1

    ls = list(d.items())
    ls.sort(key=lambda x:x[1],reverse=True)
    x_data = []
    y_data = []
    for key,values in ls[:200]:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_csv('{}-高频词Top200.csv'.format(emotion),encoding='utf-8-sig',index=False)


    # def color_func(word, font_size, position, orientation, random_state=None,
    #                **kwargs):
    #     return "hsl({}, {}%, {}%)".format(random.randint(240, 250),random.randint(20, 50),random.randint(50, 80))
    #
    # # 读取背景图片
    # background_Image = np.array(Image.open('../image.jpg'))
    # text = ' '.join(list_text)
    # wc = WordCloud(
    #     collocations=False,  # 禁用词组
    #     font_path='simhei.ttf',  # 中文字体路径
    #     margin=3,  # 词云图边缘宽度
    #     mask=background_Image,  # 背景图形
    #     scale=5,  # 放大倍数
    #     max_words=200,  # 最多词个数
    #     random_state=42,  # 随机状态
    #     width=800,  # 图片宽度
    #     height=600,  # 图片高度
    #     min_font_size=10,  # 最小字体大小
    #     max_font_size=80,  # 最大字体大小
    #     background_color='white',  # 背景颜色
    #     color_func=color_func  # 字体颜色函数
    # )
    # # 生成词云
    # wc.generate_from_text(text)
    # # 存储图像
    # wc.to_file("{}-top200-词云图.png".format(emotion))


def word1(emotion):
    new_df = pd.read_csv('{}-高频词Top200.csv'.format(emotion),encoding='gbk')
    df1 = new_df.sort_values(by=['counts'],ascending=False)
    x_data1 = list(df1['word'][:30])
    y_data1 = list(df1['counts'][:30])

    c = (
        Bar()
            .add_xaxis(x_data1)
            .add_yaxis(f"{emotion}", y_data1, label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=65)),
            title_opts=opts.TitleOpts(title=f"{emotion}-高频词Top30"))
            .render(f"{emotion}-高频词Top30.html")
    )


    data = []
    for key, values in zip(df1['word'], df1['counts']):
        data.append((str(key),int(values)))
    if emotion == 'pos':
        # 创造词云图实例时应用遮罩图片
        wordcloud = (
            WordCloud()
                .add(
                series_name="热点分析",
                data_pair=data,
                word_size_range=[8,102],
                shape='circle',  # 这里是错误的示范，正确的做法见下方
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{emotion}-top200-词云展示",
                    title_textstyle_opts=opts.TextStyleOpts(font_size=20)
                ),
                tooltip_opts=opts.TooltipOpts(is_show=True),
            )
                .render(f"{emotion}-top200-词云图.html")
        )
    else:
        # 创造词云图实例时应用遮罩图片
        wordcloud = (
            WordCloud()
                .add(
                series_name="热点分析",
                data_pair=data,
                word_size_range=[8,88],
                shape='triangle',  # 这里是错误的示范，正确的做法见下方
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{emotion}-top200-词云展示",
                    title_textstyle_opts=opts.TextStyleOpts(font_size=16)
                ),
                tooltip_opts=opts.TooltipOpts(is_show=True),
            )
                .render(f"{emotion}-top200-词云图.html")
        )


if __name__ == '__main__':
    list_emotion = ['pos','neg']
    for e in list_emotion:
        df = pd.read_excel('new_bi_lstm.xlsx')
        data = df[df['情感type'] == e]
        # word(data,e)
        word1(e)









