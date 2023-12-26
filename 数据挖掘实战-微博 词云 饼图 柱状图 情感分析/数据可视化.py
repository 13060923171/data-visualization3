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
    df = pd.read_excel('评论.xlsx')
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    new_df = df['情感分类_评论'].value_counts()
    x_data = [str(x) for x in new_df.index]
    y_data = [int(x) for x in new_df.values]

    new_df.to_csv('情感分类_评论_数据.csv',encoding='utf-8-sig')
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('情感分类')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('情感分类_评论.png')


def area_type1():
    df = pd.read_excel('评论.xlsx')

    def demo(x):
        x1 = str(x).split(" ")
        x1 = x1[0]
        if "安徽" == str(x1):
            return '安徽人'
        else:
            return "安徽以外的人"

    df['评论ip'] = df['评论ip'].apply(demo)

    df2 = df[df['情感分类_评论'] == "消极态度"]
    new_df = df2['评论ip'].value_counts()
    x_data = [str(x) for x in new_df.index]
    y_data = [int(x) for x in new_df.values]

    new_df.to_csv('消极_评论ip_数据.csv', encoding='utf-8-sig')

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('消极_地区划分')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('评论_消极_地区划分.png')


def area_type2():
    df = pd.read_excel('评论.xlsx')
    def demo(x):
        x1 = str(x).split(" ")
        x1 = x1[0]
        if "安徽" == str(x1):
            return '安徽人'
        else:
            return "安徽以外的人"

    df['评论ip'] = df['评论ip'].apply(demo)
    df1 = df[df['情感分类_评论'] == "积极态度"]

    new_df = df1['评论ip'].value_counts()
    x_data = [str(x) for x in new_df.index]
    y_data = [int(x) for x in new_df.values]

    new_df.to_csv('积极_评论ip_数据.csv', encoding='utf-8-sig')
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('积极_地区划分')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('评论_积极_地区划分.png')


def bar_1():
    df = pd.read_excel('博文.xlsx')
    df['发帖数量'] = 1
    def demo(x):
        x1 = str(x).split("-")
        x1 = x1[0]
        return x1
    df['发博时间'] = df['发博时间'].apply(demo)
    new_df = df.groupby('发博时间').agg('sum')
    new_df['发帖数量'].to_csv('每年发帖数据.csv', encoding='utf-8-sig')

    x_data = [str(x) for x in new_df['发帖数量'].index]
    y_data = [int(x) for x in new_df['发帖数量'].values]

    plt.figure(figsize=(9,6))
    plt.bar(x_data,y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('每年发帖趋势')
    plt.xlabel('年份')
    plt.ylabel('发帖数量')
    plt.savefig('每年发帖趋势.png')


def yiyuan1():
    df = pd.read_excel('评论.xlsx')
    def demo(x):
        x1 = str(x)
        if '学' in x1 or '体验' in x1 or '报名' in x1:
            return "学习意愿"
        else:
            return "其他"
    df1 = df[df['情感分类_评论'] == "积极态度"]
    df1['是否有意愿'] = df1['评论内容'].apply(demo)
    df2 = df1[df1['是否有意愿'] == "学习意愿"]
    df2.to_excel('新_评论.xlsx',index=False)


def yiyuan2():
    df = pd.read_excel('评论.xlsx')

    def demo(x):
        x1 = str(x)
        if '学' in x1 or '体验' in x1 or '报名' in x1:
            return "学习意愿"
        else:
            return "其他"

    df1 = df[df['情感分类_评论'] == "积极态度"]
    df1['是否有意愿'] = df1['评论内容'].apply(demo)
    df2 = df1[df1['是否有意愿'] == "学习意愿"]
    df2.to_excel('新_评论.xlsx', index=False)

def pie_1():
    df1 = pd.read_excel('新_博文.xlsx')
    df2 = pd.read_excel('返-新_博文.xlsx')
    bowen_len = len(df1) - len(df2)

    y_data = [bowen_len,len(df2)]
    x_data = ['其他','学习意愿']

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('博文_学习意义占比')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('博文_学习意义占比.png')


def pie_2():
    df1 = pd.read_excel('新_评论.xlsx')
    df2 = pd.read_excel('返-新_评论.xlsx')
    pinglun_len = len(df1) - len(df2)

    y_data = [pinglun_len,len(df2)]
    x_data = ['其他','学习意愿']

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('评论_学习意义占比')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('评论_学习意义占比.png')


def word_1():
    df = pd.read_excel('新_评论.xlsx')
    d = {}
    list_text = []
    for t in df['评论内容_分词']:
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

    data.to_csv('评论_高频词Top100.csv',encoding='utf-8-sig',index=False)
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
                              output_name='评论_词云图.png')
    # 开始生成图片
    Image(filename='评论_词云图.png')


def word_2():
    df = pd.read_excel('新_博文.xlsx')
    d = {}
    list_text = []
    for t in df['博文内容_分词']:
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

    data.to_csv('博文_高频词Top100.csv',encoding='utf-8-sig',index=False)
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
                              output_name='博文_词云图.png')
    # 开始生成图片
    Image(filename='博文_词云图.png')


if __name__ == '__main__':
    # emotion_type()
    # area_type1()
    # area_type2()
    # bar_1()
    # yiyuan1()
    # yiyuan2()
    # pie_1()
    # pie_2()
    word_1()
    word_2()
