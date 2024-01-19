import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib

from collections import Counter
import itertools
import jieba
import jieba.posseg as pseg

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread

def tf_idf(df,area,name):
    # corpus = []
    # for i in df['分词']:
    #     corpus.append(i.strip())
    #
    #     # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    # vectorizer = CountVectorizer()
    #
    # # 该类会统计每个词语的tf-idf权值
    # transformer = TfidfTransformer()
    #
    # # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    # tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # # 获取词袋模型中的所有词语
    # word = vectorizer.get_feature_names_out()
    #
    # # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
    # weight = tfidf.toarray()
    #
    # data = {'word': word,
    #         'tfidf': weight.sum(axis=0).tolist()}
    #
    # df2 = pd.DataFrame(data)
    # df2['tfidf'] = df2['tfidf'].astype('float64')
    # df2 = df2.sort_values(by=['tfidf'],ascending=False)
    # df2.to_csv('./需求二/{}_{}_tfidf.csv'.format(area,name),encoding='utf-8-sig',index=False)

    df2 = pd.read_csv('./需求二/{}_{}_tfidf.csv'.format(area,name))
    df2['tfidf'] = df2['tfidf'].astype('float64')
    df2 = df2.sort_values(by=['tfidf'], ascending=False)
    df2 = df2.iloc[:100]
    # 导入停用词列表
    stop_words = ['肖战', '北京', '真的','肖明','河南','链接','有限公司','哈哈哈','许凯','一种','两个','只能','秦施','杨幂','工作室','二八','视频','小说','微博','孩子','室友','姐姐','刘耀文']

    list_word = []
    for i, j in zip(df2['word'], df2['tfidf']):
        if i not in stop_words:
            list_word.append((i, int(j)))
    # c = (
    #     WordCloud()
    #         .add("{}".format(name),list_word, word_size_range=[20, 100], shape=SymbolType.DIAMOND)
    #         .set_global_opts(title_opts=opts.TitleOpts(title="{}_{}_tf-idf".format(area,name)))
    #         .render("./需求二/{}_{}_tf-idf.html".format(area,name))
    # )

    # 将词频数据转换为使用空格隔开的词汇，词频越高的词汇出现次数越多
    text = ' '.join(word for word, freq in list_word for _ in range(freq))

    # 设置中文字体
    font_path = 'C:\Windows\Fonts\simhei.ttf'  # 思源黑体
    # 读取背景图片
    background_Image = np.array(Image.open('中国地图.jpg'))
    # 提取背景图片颜色
    img_colors = ImageColorGenerator(background_Image)

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
    wc.to_file("./需求二/{}_{}_tf-idf.png".format(area,name))

if __name__ == '__main__':
    df1 = pd.read_csv('前半年数据.csv')
    df2 = pd.read_csv('后半年数据.csv')
    df3 = pd.concat([df1, df2], axis=0)
    provinces = ['安徽', '澳门', '北京', '重庆', '福建', '甘肃', '广东', '广西', '贵州', '海南', '河北', '黑龙江', '河南', '湖北', '湖南', '江苏', '江西',
                 '吉林', '辽宁', '内蒙古', '宁夏', '青海', '山东', '上海', '山西', '陕西', '四川', '台湾', '天津', '西藏', '香港', '新疆', '云南', '浙江']
    list_emotion = ['正面情感', '负面情感']
    def area(x):
        x1 = str(x)
        for p in provinces:
            if x1 in p:
                return p
    df3['IP属地'] = df3['IP属地'].apply(area)


    df3 = df3.dropna(subset=['IP属地'],axis=0)

    for p in tqdm(provinces):
        for l in list_emotion:
            df4 = df3[df3['IP属地'] == p]
            df = df4[df4['情感分类'] == l]
            try:
                tf_idf(df,p,l)
            except:
                pass





