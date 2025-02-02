import pandas as pd
import jieba.posseg as pseg
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")
import pandas as pd
import numpy as np
import random
from PIL import Image
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def main1():
    df1 = pd.read_csv('正文数据.csv')
    df2 = pd.read_csv('评论数据.csv')
    df = pd.concat([df1,df2],axis=0)
    new_df = df['情感类别'].value_counts()
    new_df.to_csv('情感分类.csv',encoding='utf-8-sig')


def main2(name):
    df1 = pd.read_csv('正文数据2.csv')
    df2 = pd.read_csv('评论数据2.csv')
    df = pd.concat([df1,df2],axis=0)
    d = {}
    list_text = []
    for t in df['fenci']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            list_text.append(i)
            d[i] = d.get(i, 0) + 1
    stop_word = []
    ls = list(d.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    for key, values in ls[:100]:
        if key not in stop_word:
            x_data.append(key)
            y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_csv(f'情感形象词频统计.csv', encoding='utf-8-sig', index=False)


def main3(name):
    # df1 = pd.read_csv('正文数据.csv')
    # df2 = pd.read_csv('评论数据.csv')
    # df = pd.concat([df1,df2],axis=0)
    # d = {}
    # list_text = []
    # for t in df['fenci']:
    #     # 把数据分开
    #     t = str(t).split(" ")
    #     for i in t:
    #         list_text.append(i)
    #         d[i] = d.get(i, 0) + 1
    #
    stop_word = []  # 你可以在这里添加停用词
    # ls = list(d.items())
    # ls.sort(key=lambda x: x[1], reverse=True)
    # x_data = []
    # y_data = []
    # z_data = []
    # for key, values in ls[:101]:
    #     if key not in stop_word:
    #         x_data.append(key)
    #         y_data.append(values)
    #         # 再次使用 jieba 获取词性
    #         word, flag = list(pseg.lcut(key))[0]
    #         z_data.append(flag)
    #
    # data = pd.DataFrame()
    # data['特征词'] = x_data
    # data['词频'] = y_data
    # data['词性'] = z_data
    #
    #
    # def tihaun(x):
    #     if x == 'Ag' or x == 'a' or x == 'ad' or x == 'an':
    #         return '形容词'
    #     elif x == 'Ng' or x == 'n' or x == 'ns' or x == 'nz' or x == 'nt':
    #         return '名词'
    #     elif x == 'v':
    #         return '动词'
    #     else:
    #         return np.NAN
    #
    # data['词性'] = data['词性'].apply(tihaun)
    # data = data.dropna(subset=['词性'],axis=0)
    # data.to_csv(f'Top100高频词.csv', encoding='utf-8-sig', index=False)

    df = pd.read_excel('相关图表.xlsx',sheet_name='Top100高频词')
    list1 = df['特征词'].tolist() + df['特征词.1'].tolist()
    list2 = df['词频'].tolist() + df['词频.1'].tolist()

    # 创建一个词频字典
    word_freq_dict = dict(zip(list1, list2))

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl({}, 100%, 50%)".format(np.random.randint(0, 300))

    # 读取背景图片
    background_Image = np.array(Image.open('images.png'))
    # text = ' '.join(list_text)
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
    # # 生成词云
    # wc.generate_from_text(text)
    # 生成词云
    wc.generate_from_frequencies(word_freq_dict)
    # 存储图像
    wc.to_file(f'{name}词云图.png')

if __name__ == '__main__':
    # main1()
    # main2('')
    main3('')