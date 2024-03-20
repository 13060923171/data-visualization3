import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

sns.set_style(style="whitegrid")

def emotion_type(df,name1,name2):
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    new_df = df['sentiment_class'].value_counts()
    x_data = [str(x) for x in new_df.index]
    y_data = [int(x) for x in new_df.values]

    new_df.to_excel('{}/{}_情感分类_数据.xlsx'.format(name1,name2))
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('sentiment_class')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('{}/{}_sentiment_class.png'.format(name1,name2))


def word_1(df,name1,name2):
    stop_words = []
    with open('常用英文停用词(NLP处理英文必备)stopwords.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip().replace("'", ""))
    d = {}
    list_text = []
    for t in df['clearn_text']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            if i not in stop_words and len(i) > 2:
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

    data.to_csv('./{}/{}_高频词Top100.csv'.format(name1,name2),encoding='utf-8-sig',index=False)

    # 设置中文字体
    font_path = 'C:\Windows\Fonts\simhei.ttf'  # 思源黑体
    # 读取背景图片
    background_Image = np.array(Image.open('中国地图.jpg'))
    # 提取背景图片颜色
    img_colors = ImageColorGenerator(background_Image)
    text = ' '.join(list_text)
    wc = WordCloud(
        stopwords=STOPWORDS.add("一个"),
        collocations=False,
        font_path=font_path,  # 中文需设置路径
        margin=1,  # 页面边缘
        mask=background_Image,
        scale=10,
        max_words=100,  # 最多词个数
        min_font_size=4,

        random_state=42,
        width=600,
        height=900,
        background_color='SlateGray',  # 背景颜色
        # background_color = '#C3481A', # 背景颜色
        max_font_size=100,

    )
    # 生成词云
    wc.generate_from_text(text)

    # 获取文本词排序，可调整 stopwords
    process_word = WordCloud.process_text(wc, text)
    sort = sorted(process_word.items(), key=lambda e: e[1], reverse=True)

    # 设置为背景色，若不想要背景图片颜色，就注释掉
    wc.recolor(color_func=img_colors)

    # 存储图像
    wc.to_file("./{}/{}_词云图.png".format(name1,name2))

def demo1(df,name1,name2):
    keyword1 = 'ChinaYummy'
    new_df = df[(df['关键字'].str.contains('{}'.format(keyword1), case=False))]
    keyword2 = 'sanxingdui'
    new_df1 = new_df[(new_df['描述'].str.contains('{}'.format(keyword2), case=False))]
    new_df1.to_excel('./{}/含有sanxingdui_{}.xlsx'.format(name1,name2),index=False)


def demo2():
    data = []
    df1 = pd.read_excel('./facebook/评论统计.xlsx')
    for d in df1['clearn_text']:
        data.append(d)
    # df2 = pd.read_excel('./facebook/all_post.xlsx')
    # for d in df2['clearn_text']:
    #     data.append(d)

    df3 = pd.read_excel('./ins/comment.xlsx')
    for d in df3['clearn_text']:
        data.append(d)
    # df4 = pd.read_excel('./ins/post_data.xlsx')
    # for d in df4['clearn_text']:
    #     data.append(d)

    # df5 = pd.read_excel('./twitter/博文表.xlsx')
    # for d in df5['clearn_text']:
    #     data.append(d)
    df6 = pd.read_excel('./twitter/评论表.xlsx')
    for d in df6['clearn_text']:
        data.append(d)

    df7 = pd.read_excel('./youtube/comment_list.xlsx')
    for d in df7['clearn_text']:
        data.append(d)
    # df8 = pd.read_excel('./youtube/post_detail.xlsx')
    # for d in df8['clearn_text']:
    #     data.append(d)

    stop_words = []
    with open('常用英文停用词(NLP处理英文必备)stopwords.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip().replace("'", ""))

    d = {}
    list_text = []
    for t in data:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            if i not in stop_words and len(i) > 2:
                # 添加到列表里面
                list_text.append(i)
                d[i] = d.get(i, 0) + 1

    ls = list(d.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    for key, values in ls[:200]:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_csv('总-高频词Top200.csv', encoding='utf-8-sig', index=False)

    # 设置中文字体
    font_path = 'C:\Windows\Fonts\simhei.ttf'  # 思源黑体
    # 读取背景图片
    background_Image = np.array(Image.open('中国地图.jpg'))
    # 提取背景图片颜色
    img_colors = ImageColorGenerator(background_Image)
    text = ' '.join(list_text)
    wc = WordCloud(
        stopwords=STOPWORDS.add("一个"),
        collocations=False,
        font_path=font_path,  # 中文需设置路径
        margin=1,  # 页面边缘
        mask=background_Image,
        scale=10,
        max_words=200,  # 最多词个数
        min_font_size=4,

        random_state=42,
        width=600,
        height=900,
        background_color='SlateGray',  # 背景颜色
        # background_color = '#C3481A', # 背景颜色
        max_font_size=100,

    )
    # 生成词云
    wc.generate_from_text(text)

    # 获取文本词排序，可调整 stopwords
    process_word = WordCloud.process_text(wc, text)
    sort = sorted(process_word.items(), key=lambda e: e[1], reverse=True)

    # 设置为背景色，若不想要背景图片颜色，就注释掉
    wc.recolor(color_func=img_colors)

    # 存储图像
    wc.to_file("总-词云图.png")


if __name__ == '__main__':
    # type_class = 'youtube'
    # demo_name = '用户视频列表'
    # csv_name = './{}/{}'.format(type_class,demo_name)
    # df = pd.read_excel('{}.xlsx'.format(csv_name))
    # emotion_type(df,type_class,demo_name)
    # word_1(df,type_class,demo_name)
    # demo1(df,type_class,demo_name)
    demo2()
