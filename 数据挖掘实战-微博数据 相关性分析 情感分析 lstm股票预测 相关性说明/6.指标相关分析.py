import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

df = pd.read_csv('./data/新_博文表.csv')

df['博文转发数'] = df['博文转发数'].fillna(0)
df['博文评论数'] = df['博文评论数'].fillna(0)
df['博文点赞数'] = df['博文点赞数'].fillna(0)

# 对每一列进行Min-Max归一化处理
df['转发数_归一化'] = (df['博文转发数'] - df['博文转发数'].min()) / (df['博文转发数'].max() - df['博文转发数'].min())
df['评论数_归一化'] = (df['博文评论数'] - df['博文评论数'].min()) / (df['博文评论数'].max() - df['博文评论数'].min())
df['点赞数_归一化'] = (df['博文点赞数'] - df['博文点赞数'].min()) / (df['博文点赞数'].max() - df['博文点赞数'].min())

# 计算互动量
df['互动量'] = df['转发数_归一化'] + df['评论数_归一化'] + df['点赞数_归一化']
df['情感热度'] = df['情感得分']

df1 = df['互动量'].describe()
df2 = df['情感热度'].describe()

# 将df1和df2的描述性统计信息保存为txt文件
with open('统计信息.txt', 'w') as f:
    f.write("互动量统计信息:\n")
    f.write(df1.to_string())
    f.write("\n\n情感热度统计信息:\n")
    f.write(df2.to_string())


def time_process1(x):
    try:
        x1 = str(x).split(" ")
        x1 = x1[0]
        if '2023年' not in x1 and '2022年' not in x1:
            x1 = '2024年' + x1
        else:
            x1 = x1
        return x1
    except:
        return np.NAN


df['博文发布时间'] = df['博文发布时间'].apply(time_process1)
df['博文发布时间'] = pd.to_datetime(df['博文发布时间'],format='%Y年%m月%d日')
df = df.sort_values(by=['博文发布时间'],ascending=True)
# 提取月份并创建一个新的列来存储月份信息
df['博文发布月份'] = df['博文发布时间'].dt.to_period('M')

# 按月份对互动量和情感热度进行聚合计算平均值
monthly_trends = df.groupby('博文发布月份').agg({'互动量': 'sum', '情感热度': 'mean'}).reset_index()

def line1():
    # 绘制互动量和情感热度的变化趋势图
    # 绘制百分比趋势柱状图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 6),dpi=500)

    plt.plot(monthly_trends['博文发布月份'].astype(str), monthly_trends['互动量'], marker='o', label='互动量')

    plt.title('互动量月度变化趋势')
    plt.xlabel('月份')
    plt.ylabel('总和')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('互动量月度变化趋势.png')
    plt.show()


def line2():
    # 绘制互动量和情感热度的变化趋势图
    # 绘制百分比趋势柱状图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 6), dpi=500)

    plt.plot(monthly_trends['博文发布月份'].astype(str), monthly_trends['情感热度'], marker='o', label='情感热度')

    plt.title('情感热度的月度变化趋势')
    plt.xlabel('月份')
    plt.ylabel('平均值')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('情感热度的月度变化趋势.png')
    plt.show()


line1()
line2()