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
    # df = pd.read_csv('new_data.csv')
    # d = {}
    # list_text = []
    # for t in df['fenci']:
    #     # 把数据分开
    #     t = str(t).split(" ")
    #     for i in t:
    #         list_text.append(i)
    #         d[i] = d.get(i, 0) + 1
    #
    # ls = list(d.items())
    # ls.sort(key=lambda x: x[1], reverse=True)
    # x_data = []
    # y_data = []
    # for key, values in ls[:100]:
    #     x_data.append(key)
    #     y_data.append(values)
    #
    # data = pd.DataFrame()
    # data['word'] = x_data
    # data['counts'] = y_data
    #
    # data.to_csv(f'高频词Top100.csv', encoding='utf-8-sig', index=False)

    # 导入停用词列表
    stop_words = []
    with open("stopwords.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    df = pd.read_csv('高频词.csv')
    # 将DataFrame转换为频率字典
    word_freq = pd.Series(df.counts.values, index=df.word).to_dict()

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl({}, 100%, 50%)".format(np.random.randint(0, 300))

    # 读取背景图片
    background_Image = np.array(Image.open('images.png'))
    # text = ' '.join(list_text)
    wc = WordCloud(
        collocations=False,  # 禁用词组
        font_path='simhei.ttf',  # 中文字体路径
        margin=3,  # 词云图边缘宽度
        stopwords=stop_words,
        mask=background_Image,  # 背景图形
        scale=3,  # 放大倍数
        max_words=300,  # 最多词个数
        random_state=42,  # 随机状态
        width=1600,  # 提高分辨率
        height=1200,
        min_font_size=10,  # 调大最小字体
        max_font_size=80,  # 调大最大字体
        background_color='white',  # 背景颜色
        color_func=color_func  # 字体颜色函数
    )
    # 生成词云
    # 直接从词频生成词云
    wc.generate_from_frequencies(word_freq)
    # 保存高清图片
    wc.to_file('custom_wordcloud.png')

if __name__ == '__main__':
    main1()