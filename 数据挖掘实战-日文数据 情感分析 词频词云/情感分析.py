import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

import random
from PIL import Image
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def emotion_pie(df):
    new_df = df['label'].value_counts()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(9, 6), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title(f'Emotional distribution')
    plt.tight_layout()
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.savefig(f'Emotional distribution.png')


def kmeans(df):
    # 导入停用词列表
    stop_words = []
    with open("stopwords-ja.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    d = {}
    list_text = []
    for t in df['processed_content']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            if i not in stop_words:
                list_text.append(i)
                d[i] = d.get(i, 0) + 1

    ls = list(d.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    for key, values in ls[:100]:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_csv(f'高频词Top100.csv', encoding='utf-8-sig', index=False)

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl({}, 100%, 50%)".format(np.random.randint(0, 300))

    # 读取背景图片
    background_Image = np.array(Image.open('images.png'))
    text = ' '.join(list_text)
    wc = WordCloud(
        collocations=False,  # 禁用词组
        font_path='simhei.ttf',  # 中文字体路径
        margin=3,  # 词云图边缘宽度
        mask=background_Image,  # 背景图形
        scale=3,  # 放大倍数
        max_words=100,  # 最多词个数
        random_state=42,  # 随机状态
        width=800,  # 图片宽度
        height=600,  # 图片高度
        min_font_size=15,  # 最小字体大小
        max_font_size=90,  # 最大字体大小
        background_color='#fdfefe',  # 背景颜色
        color_func=color_func  # 字体颜色函数
    )
    # 生成词云
    wc.generate_from_text(text)
    # 存储图像
    wc.to_file(f'词云图.png')



if __name__ == '__main__':
    df = pd.read_csv('new_processed_data.csv')
    emotion_pie(df)
    kmeans(df)



