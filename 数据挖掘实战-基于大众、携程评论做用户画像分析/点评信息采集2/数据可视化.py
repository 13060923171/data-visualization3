import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import stylecloud
from IPython.display import Image

sns.set_style(style="whitegrid")


def emotion_type():
    new_df = pd.read_excel('新-数据.xlsx')
    df = new_df[new_df['景点'] == "布达拉宫"]
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    new_df = df['情感分类_博文'].value_counts()
    x_data = [str(x) for x in new_df.index]
    y_data = [int(x) for x in new_df.values]

    new_df.to_csv('情感分类_博文_布达拉宫.csv',encoding='utf-8-sig')
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('情感分类')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('情感分类_博文_布达拉宫.png')


def bar_1():
    new_df = pd.read_excel('新-数据.xlsx')
    df = new_df[new_df['景点'] == "布达拉宫"]
    df['发帖数量'] = 1
    def demo(x):
        x1 = str(x).split("T")
        x1 = x1[0]
        return x1
    df['评论时间'] = df['评论时间'].apply(demo)
    df['评论时间'] = pd.to_datetime(df['评论时间'])
    df.index = df['评论时间']
    df_month = df['发帖数量'].resample('A-DEC').sum()

    x_data = [str(x).split(" ")[0] for x in df_month.index]
    y_data = [int(x) for x in df_month.values]

    plt.figure(figsize=(9,6))
    plt.bar(x_data,y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('每年发帖趋势')
    plt.xticks(rotation=45)  # 设置倾斜角度为45度
    plt.xlabel('年份')
    plt.ylabel('发帖数量')
    plt.savefig('布达拉宫_每年发帖趋势.png')


def word_1():
    new_df = pd.read_excel('新-数据.xlsx')
    df = new_df[new_df['景点'] == "布达拉宫"]
    d = {}
    list_text = []
    for t in df['评论_分词']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            # 对文本进行分词和词性标注
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

    data.to_csv('布达拉宫_高频词Top100.csv',encoding='utf-8-sig',index=False)
    # 然后传入词云图中，筛选最多的100个词
    stylecloud.gen_stylecloud(text=' '.join(list_text), max_words=100,
                              # 不能有重复词
                              collocations=False,
                              max_font_size=400,
                              # 字体样式
                              font_path='simhei.ttf',
                              # 图片形状
                              icon_name='fas fa-circle',
                              # 图片大小
                              size=1200,
                              # palette='matplotlib.Inferno_9',
                              # 输出图片的名称和位置
                              output_name='布达拉宫_词云图.png')
    # 开始生成图片
    Image(filename='布达拉宫_词云图.png')


if __name__ == '__main__':
    emotion_type()
    bar_1()
    word_1()

