import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

import re
import numpy as np
import pandas as pd


import random
from PIL import Image
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def img1():
    df = pd.read_excel('new_post.xlsx')
    df['情感类别'] = df['情感类别'].replace("LABEL_0","消极").replace("LABEL_1","中立").replace("LABEL_2","积极")
    # 情感分析饼图
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    df['情感类别'].value_counts().plot.pie(
        autopct='%1.1f%%',
        colors=['#e3716e', '#eca680', '#7ac7e2'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    plt.title(f'情感分布分析', fontsize=14, pad=20)
    plt.savefig(f'./图片/情感分布分析.png', dpi=300)


def img2():
    df = pd.read_excel('new_post.xlsx')
    d = {}
    list_text = []
    for t in df['分词']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            list_text.append(i)
            d[i] = d.get(i, 0) + 1

    ls = list(d.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    for key, values in ls[:300]:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    word_image(data)


def word_image(df):
        # 将DataFrame转换为频率字典
        word_freq = pd.Series(df.counts.values, index=df.word).to_dict()

        def color_func(word, font_size, position, orientation, random_state=None,
                       **kwargs):
            return "hsl({}, 100%, 50%)".format(np.random.randint(0, 300))

        # 读取背景图片
        background_Image = np.array(Image.open('images.png'))
        wc = WordCloud(
            collocations=False,  # 禁用词组
            font_path='simhei.ttf',  # 中文字体路径
            margin=3,  # 词云图边缘宽度
            mask=background_Image,  # 背景图形
            scale=3,  # 放大倍数
            max_words=150,  # 最多词个数
            random_state=42,  # 随机状态
            width=800,  # 提高分辨率
            height=600,
            min_font_size=20,  # 调大最小字体
            max_font_size=100,  # 调大最大字体
            background_color='white',  # 背景颜色
            color_func=color_func  # 字体颜色函数
        )
        # 生成词云
        # 直接从词频生成词云
        wc.generate_from_frequencies(word_freq)
        # 保存高清图片
        wc.to_file(f'./图片/词云图.png')


def img3():
    # 读取数据
    df = pd.read_excel('new_post.xlsx')

    # 替换情感类别标签（保持可读性）
    df['情感类别'] = df['情感类别'].replace({
        "LABEL_0": "消极",
        "LABEL_1": "中立",
        "LABEL_2": "积极"
    })

    # 处理时间列（保留日期，转换为datetime格式）
    df['发布时间'] = pd.to_datetime(df['发布时间']).dt.normalize()  # 去除时间部分，只留日期

    # 按**3天周期**分组（核心修正：使用'3D'划分）
    df['3天周期'] = df['发布时间'].dt.to_period('3D')  # 每个周期为连续3天（如2024-07-01~2024-07-03）

    # 统计每3天各情感类别的数量（确保所有类别存在，缺失值填0）
    result = (
        df.groupby(['3天周期', '情感类别'])
            .size()  # 计算每组数量（即该情感类别在该3天周期内的出现次数）
            .unstack(fill_value=0)  # 将情感类别转为列（列名：消极、中立、积极）
            .reindex(columns=['消极', '中立', '积极'], fill_value=0)  # 确保列顺序一致（避免乱序）
            .reset_index()  # 将"3天周期"从索引转为列（方便后续处理）
    )

    # 将3天周期的Period对象转为**起始日期**（方便绘图时x轴显示）
    # 例如：2024-07-01~2024-07-03的Period对象转为2024-07-01的Timestamp
    result['周期起始日期'] = result['3天周期'].dt.to_timestamp()
    result = result.iloc[:-1]
    # 设置中文显示（避免乱码）
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表（调整尺寸以适应更多3天周期）
    plt.figure(figsize=(20, 8))  # 宽度加大，避免x轴标签重叠

    # 定义柱状图核心参数（关键优化：解决柱子重叠问题）
    bar_width = 0.2  # 柱子宽度（单位：天），2天的宽度刚好占据3天周期的大部分空间，且不重叠
    offset1 = bar_width * 1.0  # 偏移量（用于并排显示三个情感类别的柱子）
    offset2 = bar_width * 2.0  # 偏移量（用于并排显示三个情感类别的柱子）
    # 绘制**并排柱状图**（每个3天周期内显示消极、中立、积极三个柱子）
    # 消极：向左偏移offset（占据周期起始日期的前2天）
    plt.bar(
        result['周期起始日期'] - pd.Timedelta(days=offset1),  # x坐标：周期起始日期 - 偏移量
        result['消极'],  # y坐标：消极情感数量
        width=bar_width,  # 柱子宽度
        label='消极',  # 图例标签
        alpha=0.8,  # 透明度（避免颜色过深）
        color='#1f77b4'  # 蓝色（经典配色，区分度高）
    )
    # 中立：不偏移（占据周期起始日期的中间2天）
    plt.bar(
        result['周期起始日期'],  # x坐标：周期起始日期
        result['中立'],  # y坐标：中立情感数量
        width=bar_width,
        label='中立',
        alpha=0.8,
        color='#ff7f0e'  # 橙色
    )
    # 积极：向右偏移offset（占据周期起始日期的后2天）
    plt.bar(
        result['周期起始日期'] + pd.Timedelta(days=offset1),  # x坐标：周期起始日期 + 偏移量
        result['积极'],  # y坐标：积极情感数量
        width=bar_width,
        label='积极',
        alpha=0.8,
        color='#2ca02c'  # 绿色
    )


    # 配置图表标题与坐标轴（符合3天周期主题）
    plt.title('情感类别分布走势', fontsize=16, pad=20)  # 标题：明确时间维度
    plt.xlabel('日期（每3天周期）', fontsize=12, labelpad=15)  # x轴标签：说明周期类型
    plt.ylabel('数量', fontsize=12, labelpad=15)  # y轴标签：保持简洁

    # 设置**x轴时间格式**（关键修正：显示3天周期的起始日期）
    ax = plt.gca()  # 获取当前坐标轴对象
    # 1. 设置x轴刻度为**每3天一个**（与数据分组一致）
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))  # 每3天显示一个刻度
    # 2. 设置x轴标签格式为**年-月-日**（清晰显示日期）
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # 3. 旋转x轴标签（避免重叠）
    plt.xticks(rotation=45, ha='right')  # 旋转45度，右对齐

    # 设置**y轴为整数刻度**（避免小数，符合数量逻辑）
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # 调整y轴范围（避免总数标签超出图表）
    plt.ylim(0, result[['消极', '中立', '积极']].sum(axis=1).max() + 100)  # 最大值+20，给总数标签留空间

    # 添加**网格线**（增强可读性）
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 仅显示y轴网格线（横向）

    # 添加**图例**（调整位置，避免遮挡数据）
    plt.legend(
        title='情感类别',  # 图例标题
        title_fontsize=11,  # 标题字体大小
        loc='upper left',  # 图例位置：左上角
        bbox_to_anchor=(0.02, 0.98)  # 微调图例位置（距离左边2%，距离顶部2%）
    )

    # 调整**布局**（避免元素重叠）
    plt.tight_layout()  # 自动调整子图参数，使元素不重叠

    # 保存图表（高分辨率，保留完整元素）
    plt.savefig('./图片/情感走势.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight'：去除多余空白

    # 显示图表（查看效果）
    plt.show()


def img4():
    df = pd.read_csv('./总（评论、转发）/data.csv')
    df['节点数量'] = 1
    new_df = df.groupby(by='modularity_class')['节点数量'].agg('sum')
    new_df = new_df.sort_values(ascending=False)
    new_df = new_df.iloc[:10]
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    z_data = [f"子群{z}" for z in range(1, 11)]
    x_data2 = [f"{x2}({x1})" for x1, x2 in zip(x_data, z_data)]  # 组合标签

    plt.figure(figsize=(16, 12))
    bars = plt.bar(x_data2, y_data, color='skyblue', edgecolor='black')  # 增加颜色和边框

    # ====== 核心修改：添加数值标签 ======
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,  # X坐标：柱子中心
            height + max(y_data) * 0.01,  # Y坐标：略高于柱顶
            f'{height:,}',  # 千位分隔符格式
            ha='center',  # 水平居中
            va='bottom',  # 垂直底部对齐
            fontsize=11,  # 字体大小
            fontweight='bold'  # 粗体显示
        )


    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("节点数量规模分布TOP10", fontsize=18, pad=20)
    plt.xlabel("模块化分类\n(Modularity Class)", fontsize=14, labelpad=15)
    plt.ylabel("规模（节点数量）", fontsize=14, labelpad=15)

    # 添加网格线（Y轴方向）
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 调整底部空白区域防止标签被裁剪
    plt.subplots_adjust(bottom=0.25)

    plt.savefig('./图片/节点数量规模分布top10.png', dpi=300, bbox_inches='tight')
    plt.show()

def data_process1():
    df = pd.read_csv('./总（评论、转发）/data.csv')
    df1 = df.sort_values(by=['degree'],ascending=False)
    df1['排名'] = [x+1 for x in range(len(df1))]
    data = pd.DataFrame()
    data['节点账号'] = df1['Id']
    data['所属子群'] = df1['modularity_class']
    data['点度中心性排名(值)'] = [f"{x1}({round(x2,2)})" for x1,x2 in zip(df1['排名'],df1['degree'])]
    data = data.iloc[:100]
    data.to_excel('./总（评论、转发）/点度中心性排名表.xlsx',index=False)

def data_process2():
    df = pd.read_csv('./总（评论、转发）/data.csv')
    df['closnesscentrality'] = df['closnesscentrality'].astype('float')
    df1 = df.sort_values(by=['closnesscentrality'],ascending=False)
    df1['排名'] = [x+1 for x in range(len(df1))]
    data = pd.DataFrame()
    data['节点账号'] = df1['Id']
    data['所属子群'] = df1['modularity_class']
    data['接近度中心性排名(值)'] = [f"{x1}({x2})" for x1,x2 in zip(df1['排名'],df1['closnesscentrality'])]
    data = data.iloc[:100]
    data.to_excel('./总（评论、转发）/接近度中心性排名表.xlsx',index=False)

def data_process3():
    df = pd.read_csv('./总（评论、转发）/data.csv')
    df1 = df.sort_values(by=['betweenesscentrality'],ascending=False)
    df1['排名'] = [x+1 for x in range(len(df1))]
    data = pd.DataFrame()
    data['节点账号'] = df1['Id']
    data['所属子群'] = df1['modularity_class']
    data['中介中心性排名(值)'] = [f"{x1}({round(x2,2)})" for x1,x2 in zip(df1['排名'],df1['betweenesscentrality'])]
    data = data.iloc[:100]
    data.to_excel('./总（评论、转发）/中介中心性排名表.xlsx',index=False)

if __name__ == '__main__':
    # img1()
    # img2()
    img3()
    # img4()
    data_process1()
    data_process2()
    data_process3()

