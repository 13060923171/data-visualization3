import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

import random
from PIL import Image
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.preprocessing import StandardScaler

from pyecharts.charts import Geo
from pyecharts import options as opts

df = pd.read_csv('data.csv')

def datetime1(x):
    x = str(x).replace('-','/').split(" ")
    return x[0]

df['日期'] = df['日期'].apply(datetime1)
# 如果日期列是字符串，使用pd.to_datetime转换，并指定格式
date_format = "%Y/%m/%d"
df.index = pd.to_datetime(df['日期'], format=date_format)
# 按日期排序
df = df.sort_index()


def line1(name,name1):
    # 使用resample方法对数据进行重采样，并使用相应的聚合方法
    monthly_data = df['{}'.format(name)].resample('M').sum()
    monthly_data.to_csv('./data/{}_时间数据.csv'.format(name),encoding='utf-8-sig')
    x_data = [x for x in monthly_data.index]
    y_data = [y for y in monthly_data.values]

    plt.figure(figsize=(16, 9), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 绘制线形图
    # 绘制线形图
    plt.plot(x_data, y_data, label='每月{}总数'.format(name1), color='#D98880', marker='o', linestyle='-')

    plt.xlabel('月份')
    plt.ylabel('数量')
    plt.xticks(rotation=75)
    # 保证 x 轴显示所有月份
    plt.title('每月{}总数趋势图'.format(name1))
    plt.legend()
    plt.grid(True)
    # 显示图例
    plt.savefig('./img/每月{}总数趋势图.png'.format(name1))
    plt.show()


def line2(area,df,name,name1):
    # 使用resample方法对数据进行重采样，并使用相应的聚合方法
    monthly_data = df['{}'.format(name)].resample('M').sum()
    monthly_data.to_csv('./data/{}-{}_时间数据.csv'.format(area,name),encoding='utf-8-sig')
    x_data = [x for x in monthly_data.index]
    y_data = [y for y in monthly_data.values]

    plt.figure(figsize=(16, 9), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 绘制线形图
    # 绘制线形图
    plt.plot(x_data, y_data, label='{}-每月{}总数'.format(area,name1), color='#85C1E9', marker='o', linestyle='-')

    plt.xlabel('月份')
    plt.ylabel('数量')
    plt.xticks(rotation=75)
    # 保证 x 轴显示所有月份
    plt.title('{}-每月{}总数趋势图'.format(area,name1))
    plt.legend()
    plt.grid(True)
    # 显示图例
    plt.savefig('./img/{}-每月{}总数趋势图.png'.format(area,name1))
    plt.show()


def word(df,name):
    d = {}
    list_text = []
    for t in df['fenci']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            # 添加到列表里面
            list_text.append(i)
            d[i] = d.get(i,0)+1

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl({}, {}%, {}%)".format(random.randint(240, 250),random.randint(20, 50),random.randint(50, 80))

    # 读取背景图片
    background_Image = np.array(Image.open('image.jpg'))
    text = ' '.join(list_text)
    wc = WordCloud(
        collocations=False,  # 禁用词组
        font_path='simhei.ttf',  # 中文字体路径
        margin=3,  # 词云图边缘宽度
        mask=background_Image,  # 背景图形
        scale=5,  # 放大倍数
        max_words=100,  # 最多词个数
        random_state=42,  # 随机状态
        width=800,  # 图片宽度
        height=600,  # 图片高度
        min_font_size=10,  # 最小字体大小
        max_font_size=80,  # 最大字体大小
        background_color='white',  # 背景颜色
        color_func=color_func  # 字体颜色函数
    )

    # 生成词云
    wc.generate_from_text(text)
    # 存储图像
    wc.to_file("./img/主题-{}-top100-词云图.png".format(name))


def bar1(df):
    def datetime1(x):
        x = str(x).replace('-', '/').split(" ")
        return x[0]

    df['日期'] = df['日期'].apply(datetime1)
    # 如果日期列是字符串，使用pd.to_datetime转换，并指定格式
    date_format = "%Y/%m/%d"
    df.index = pd.to_datetime(df['日期'], format=date_format)
    # 按日期排序
    df = df.sort_index()

    data = df[['转发数','评论数','点赞数','阅读数']]
    # 使用StandardScaler对数据进行标准化
    scaler = StandardScaler()
    data_score = scaler.fit_transform(data)
    score = []
    for i in data_score:
        zongliang = i[0] + i[1] + i[2] + i[3]
        score.append(zongliang)
    df['总体热度'] = score

    # 使用resample方法对数据进行重采样，并使用相应的聚合方法
    monthly_data = df['总体热度'].resample('M').sum()
    monthly_data.to_csv('./data/总体热度_时间数据.csv', encoding='utf-8-sig')
    x_data = [x for x in monthly_data.index]
    y_data = [y for y in monthly_data.values]

    plt.figure(figsize=(16, 9), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制线形图
    # 绘制线形图
    plt.bar(x_data, y_data, label='月均总体热度', color='#D98880',width=3)

    plt.xlabel('月份')
    plt.ylabel('数量')
    plt.xticks(rotation=75)
    # 保证 x 轴显示所有月份
    plt.title('月均总体热度变化趋势')
    plt.legend()
    plt.grid(True)
    # 显示图例
    plt.savefig('./img/月均总体热度变化趋势图.png')
    plt.show()


def line3(df):
    # 使用resample方法对数据进行重采样，并使用相应的聚合方法
    monthly_data = df['sentiment score'].resample('M').mean()
    monthly_data.to_csv('./data/情感分析_时间数据.csv',encoding='utf-8-sig')
    x_data = [x for x in monthly_data.index]
    y_data = [y for y in monthly_data.values]

    plt.figure(figsize=(16, 9), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 绘制线形图
    # 绘制线形图
    plt.plot(x_data, y_data, label='月均情感', color='red', marker='o', linestyle='-')

    plt.xlabel('月份')
    plt.ylabel('数量')
    plt.xticks(rotation=75)
    # 保证 x 轴显示所有月份
    plt.title('月均情感分值变化趋势')
    plt.legend()
    plt.grid(True)
    # 显示图例
    plt.savefig('./img/月均情感分值变化趋势图.png')
    plt.show()


def map_analyze(df):
    data1 = pd.DataFrame()
    data1['信源地域'] = df['信源地域']
    data1['sentiment score'] = df['sentiment score']
    new_df = data1.groupby('信源地域').agg('mean')
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]

    data_list = []
    for x,y in zip(x_data,y_data):
        if x != '其他' and x != '境外':
            data_list.append((x,y))
    c = (
        Geo()
            .add_schema(maptype="china")
            .add(" ", data_list)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(
                is_piecewise=True,
                pieces=[
                    {"min": 0.65, "max": 0.75, "label": "一般", "color": "#000000"},
                    {"min": 0.75, "max": 0.85, "label": "积极", "color": "#808080"},
                    {"min": 0.85, "max": 1, "label": "较为积极", "color": "#FFFFFF"}
                ]
            ),
            title_opts=opts.TitleOpts(title="地域公众情感差异分析"),
        )
            .render("地域公众情感差异分析.html")
    )


if __name__ == '__main__':
    list_name = ['转发数','评论数','点赞数']
    list_name1 = ['转发','评论','点赞']
    for n1,n2 in zip(list_name,list_name1):
        line1(n1,n2)

    area_df = df['信源地域'].value_counts()

    x_data1 = [x for x in area_df.index]

    for x in x_data1:
        df1 = df[df['信源地域'] == x]
        for n1, n2 in zip(list_name, list_name1):
            line2(x,df1,n1,n2)

    data = pd.read_csv('./LDA/lda_data.csv')
    new_df = data['主题类型'].value_counts()
    x_data2 = [x for x in new_df.index]
    for x in x_data2:
        data1 = data[data['主题类型'] == x]
        word(data1,x)
    bar1(data)
    line3(data)

    map_analyze(data)


