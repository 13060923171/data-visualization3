import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
import stylecloud


df = pd.read_csv('聚类结果.csv')


def main1():
    df1 = df[df['聚类结果'] == 1]
    score = df1['comp_score'].value_counts()
    x_data = list(score.index)
    y_data = list(score.values)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6))  # 调节图形大小
    labels = x_data  # 定义标签
    sizes = y_data  # 每块值
    colors = ['#2E86C1', '#5DADE2','#EC7063']  # 每块颜色定义
    explode = (0, 0.05,0)  # 将某一块分割出来，值越大分割出的间隙越大
    patches, text1, text2 = plt.pie(sizes,
                                    explode=explode,
                                    labels=labels,
                                    colors=colors,
                                    labeldistance=1.1,  # 图例距圆心半径倍距离
                                    autopct='%3.2f%%',  # 数值保留固定小数位
                                    shadow=False,  # 无阴影设置
                                    startangle=90,  # 逆时针起始角度设置
                                    pctdistance=0.6)  # 数值距圆心半径倍数距离
    # patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
    # x，y轴刻度设置一致，保证饼图为圆形
    plt.axis('equal')
    plt.legend()
    plt.title('聚类1-情感占比分析')
    plt.savefig('聚类1-情感占比分析.png')
    plt.show()

    stop_words = []
    list_text = []
    for t in df1['comment']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            # 再过滤一遍无效词
            if i not in stop_words:
                # 添加到列表里面
                list_text.append(i)
    # 然后传入词云图中，筛选最多的100个词
    stylecloud.gen_stylecloud(text=' '.join(list_text), max_words=100,
                              # 不能有重复词
                              collocations=False,
                              # 字体样式
                              font_path='simhei.ttf',
                              # 图片形状
                              icon_name='fas fa-socks',
                              # 图片大小
                              size=800,
                              # palette='matplotlib.Inferno_9',
                              # 输出图片的名称和位置
                              output_name='聚类1-词云图.png')
    # 开始生成图片
    Image(filename='聚类1-词云图.png')


def main0():
    df1 = df[df['聚类结果'] == 0]
    score = df1['comp_score'].value_counts()
    x_data = list(score.index)
    y_data = list(score.values)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6))  # 调节图形大小
    labels = x_data  # 定义标签
    sizes = y_data  # 每块值
    colors = ['#2E86C1', '#5DADE2','#EC7063']  # 每块颜色定义
    explode = (0, 0.05,0)  # 将某一块分割出来，值越大分割出的间隙越大
    patches, text1, text2 = plt.pie(sizes,
                                    explode=explode,
                                    labels=labels,
                                    colors=colors,
                                    labeldistance=1.1,  # 图例距圆心半径倍距离
                                    autopct='%3.2f%%',  # 数值保留固定小数位
                                    shadow=False,  # 无阴影设置
                                    startangle=90,  # 逆时针起始角度设置
                                    pctdistance=0.6)  # 数值距圆心半径倍数距离
    # patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
    # x，y轴刻度设置一致，保证饼图为圆形
    plt.axis('equal')
    plt.legend()
    plt.title('聚类0-情感占比分析')
    plt.savefig('聚类0-情感占比分析.png')
    plt.show()

    stop_words = []
    list_text = []
    for t in df1['comment']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            # 再过滤一遍无效词
            if i not in stop_words:
                # 添加到列表里面
                list_text.append(i)
    # 然后传入词云图中，筛选最多的100个词
    stylecloud.gen_stylecloud(text=' '.join(list_text), max_words=100,
                              # 不能有重复词
                              collocations=False,
                              # 字体样式
                              font_path='simhei.ttf',
                              # 图片形状
                              icon_name='fas fa-crown',
                              # 图片大小
                              size=800,
                              # palette='matplotlib.Inferno_9',
                              # 输出图片的名称和位置
                              output_name='聚类0-词云图.png')
    # 开始生成图片
    Image(filename='聚类0-词云图.png')


main1()
main0()