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
    df = pd.read_excel('model_metrics.xlsx')

    Accuracy = df['Accuracy'].tolist()
    Precision = df['Precision'].tolist()
    Recall = df['Recall'].tolist()
    F1_Score = df['F1 Score'].tolist()

    name = ['nb_count','nb_tfidf','lstm','gru']


    # 创建柱状图的函数
    def create_bar_chart(data, title, metric_name):
        plt.figure(figsize=(10, 6),dpi=500)
        sns.barplot(x=name, y=data, palette='viridis')
        plt.title(title, fontsize=16)
        plt.ylabel(metric_name, fontsize=14)
        plt.xlabel('Model', fontsize=14)

        # 显示数值
        for index, value in enumerate(data):
            plt.text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

        plt.savefig(f'{title}.png')
        plt.show()


    # 创建 Accuracy 柱状图
    create_bar_chart(Accuracy, 'Accuracy Comparison', 'Accuracy')

    # 创建 Precision 柱状图
    create_bar_chart(Precision, 'Precision Comparison', 'Precision')

    # 创建 Recall 柱状图
    create_bar_chart(Recall, 'Recall Comparison', 'Recall')

    # 创建 F1 Score 柱状图
    create_bar_chart(F1_Score, 'F1 Score Comparison', 'F1 Score')


def main2(name):
    df = pd.read_csv('new_data.csv')
    # df = df1[df1['label'] == 0]
    d = {}
    list_text = []
    for t in df['fenci']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            list_text.append(i)
            d[i] = d.get(i, 0) + 1
    # stop_word = ['好吃','不错','喜欢','挺好吃','满意','挺快']
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

    data.to_csv(f'{name}-高频词.csv', encoding='utf-8-sig', index=False)

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl({}, 100%, 50%)".format(np.random.randint(0, 300))

    # 读取背景图片
    background_Image = np.array(Image.open('images.png'))
    text = ' '.join(list_text)
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
    # 生成词云
    wc.generate_from_text(text)
    # 存储图像
    wc.to_file(f'{name}-词云图.png')


def emotion_pie():
    df1 = pd.read_csv('new_data.csv')
    new_df = df1['label'].value_counts()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    x_data1 = ['负面','正面']
    plt.figure(figsize=(9, 6), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data1, startangle=0, autopct='%1.2f%%')
    plt.title('情感占比分布')
    plt.tight_layout()
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.savefig('情感占比分布.png')

if __name__ == '__main__':
    # main1()
    # main2('整体')
    emotion_pie()