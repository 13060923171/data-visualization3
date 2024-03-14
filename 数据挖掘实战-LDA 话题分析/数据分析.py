import pandas as pd
import os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def class_topic(data,name):
    if not os.path.exists("./{}".format(name)):
        os.mkdir("./{}".format(name))

    d = {}
    list_text = []
    for t in data['fenci']:
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

    data.to_csv('./{}/高频词Top100.csv'.format(name),encoding='utf-8-sig',index=False)
    # 将词频数据转换为使用空格隔开的词汇，词频越高的词汇出现次数越多

    # 读取背景图片
    background_Image = np.array(Image.open('image.jpg'))
    # 提取背景图片颜色
    img_colors = ImageColorGenerator(background_Image)
    text = ' '.join(list_text)
    wc = WordCloud(
        collocations=False,
        font_path='simhei.ttf',
        margin=1,  # 页面边缘
        mask=background_Image,
        scale=10,
        max_words=100,  # 最多词个数
        random_state=42,
        width=900,
        height=600,
        background_color='SlateGray',  # 背景颜色
        # background_color = '#C3481A', # 背景颜色

    )
    # 生成词云
    wc.generate_from_text(text)

    # 设置为背景色，若不想要背景图片颜色，就注释掉
    wc.recolor(color_func=img_colors)

    # 存储图像
    wc.to_file("./{}/词云图.png".format(name))

def time_post1(df,name):
    df['创建时间'] = pd.to_datetime(df['创建时间'])
    df.index = df['创建时间']
    df['发帖数量'] = 1
    new_df = df['发帖数量'].resample('M').sum()
    new_df = new_df.sort_index()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x_data,y_data, color='#b82410', label='{}'.format(name))
    plt.legend()
    plt.title('每月发帖变化趋势')
    plt.xlabel('Month')
    plt.ylabel('Values')
    plt.grid()
    plt.savefig('./{}/每月发帖变化趋势.png'.format(name))
    new_df.to_csv('./{}/每月发帖变化趋势数据.csv'.format(name),encoding='utf-8-sig',index=False)

def time_post2(df,name):
    df['创建时间'] = pd.to_datetime(df['创建时间'])
    df.index = df['创建时间']
    df['发帖数量'] = 1
    new_df = df['发帖数量'].resample('Q').sum()
    new_df = new_df.sort_index()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x_data,y_data, color='#b82410', label='{}'.format(name))
    plt.legend()
    plt.title('每季度发帖变化趋势')
    plt.xlabel('Quarter')
    plt.ylabel('Values')
    plt.grid()
    plt.savefig('./{}/每季发帖变化趋势.png'.format(name))
    new_df.to_csv('./{}/每季发帖变化趋势数据.csv'.format(name),encoding='utf-8-sig',index=False)


def main1():
    df = pd.read_csv('./LDA/lda_data.csv')
    new_df = df['主题类型'].value_counts()
    x_data = [x for x in new_df.index]
    topic = x_data[0]
    data = df[df['主题类型'] == topic]
    name = '热门主题'
    class_topic(data, name)
    time_post1(data, name)
    time_post2(data, name)

def main2():
    df = pd.read_csv('./LDA/lda_data.csv')

    def demo(x):
        x1 = int(x)
        if x1 == 5:
            return '恋爱'
        if x1 == 4 or x1 == 7:
            return '学习'
        if x1 == 0 or x1 == 8:
            return '校园生活'
        if x1 == 1 or x1 == 3:
            return '工作'
        if x1 == 6 or x1 == 2 or x1 == 9:
            return '其他'

    df['主题类型'] = df['主题类型'].apply(demo)
    new_df = df['主题类型'].value_counts()
    x_data = [x for x in new_df.index]
    for x in x_data:
        data = df[df['主题类型'] == x]
        class_topic(data, x)
        time_post1(data, x)
        time_post2(data, x)

    def demo2(x):
        x1 = str(x).split("-")
        x1 = x1[0]
        return x1

    df['创建时间'] = df['创建时间'].apply(demo2)
    new_df1 = df['创建时间'].value_counts()
    new_df1 = new_df1.sort_index()
    x_data1 = [x for x in new_df1.index]
    for x in x_data1:
        data = df[df['创建时间'] == x]
        plt.style.use('ggplot')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure(figsize=(9,6),dpi=500)
        new_df2 = data['主题类型'].value_counts()

        x_data = [x for x in new_df2.index]
        y_data = [x for x in new_df2.values]
        plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
        plt.title('{} theme strength'.format(x))
        plt.legend(x_data, loc='lower right')
        plt.tight_layout()
        plt.savefig('{} theme strength.png'.format(x))

if __name__ == '__main__':
    main1()
    main2()


