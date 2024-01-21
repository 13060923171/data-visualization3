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
    df = pd.read_excel('data.xlsx')
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    new_df = df['sentiment_class'].value_counts()
    x_data = [str(x) for x in new_df.index]
    y_data = [int(x) for x in new_df.values]

    new_df.to_csv('情感分类_数据.csv',encoding='utf-8-sig')
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('sentiment_class')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('sentiment_class.png')


def word_1():
    df = pd.read_excel('data.xlsx')

    stop_words = ['ccp', 'br', 'de', 'la', 'en', 'da', 'amp', 'el', 'por', 'du', 'je', 'cha', 'don', 'bro', 'se', 'gz',
                  'ich', 'uk', 'soo', 'ppl', 'ho', 'st', 'di', 'ist', 'siu']
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

    data.to_csv('高频词Top100.csv',encoding='utf-8-sig',index=False)
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
                              output_name='词云图.png')
    # 开始生成图片
    Image(filename='词云图.png')


def values_shiping():
    df = pd.read_excel('data.xlsx')
    mean_values = []
    std_values = []
    number_values = []
    for i in range(1,61):
        try:
            df1 = df[df['视频编号'] == i]
            mean = df1['compound'].mean()
            std = df1['compound'].std()
            mean_values.append(mean)
            std_values.append(std)
            number_values.append(i)
        except:
            pass

    data = pd.DataFrame()
    data['视频编号'] = number_values
    data['情感分值的平均值'] = mean_values
    data['情感分值的标准差'] = mean_values

    data.to_excel('情感得分相关数据.xlsx')


if __name__ == '__main__':
    emotion_type()
    word_1()
    values_shiping()