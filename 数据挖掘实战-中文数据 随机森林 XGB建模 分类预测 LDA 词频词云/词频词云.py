import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

sns.set_style(style="whitegrid")

def word_1(df,sentiment):
    d = {}
    list_text = []
    for t in df['new_content']:
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

    data.to_csv('./LDA/{}/高频词Top100.csv'.format(sentiment),encoding='utf-8-sig',index=False)
    # 将词频数据转换为使用空格隔开的词汇，词频越高的词汇出现次数越多


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
    wc.to_file("./LDA/{}/词云图.png".format(sentiment))


if __name__ == '__main__':
    list_sentiment = ['消极', '较为消极', "中立", "积极", "较为积极"]
    df = pd.read_csv('new_data.csv')
    # for s in list_sentiment:
    #     df2 = df[df['情感极性'] == s]
    #     word_1(df2, s)

    word_1(df,"总体")

