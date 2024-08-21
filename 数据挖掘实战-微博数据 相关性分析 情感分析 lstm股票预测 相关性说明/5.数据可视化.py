import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

df = pd.read_csv('./data/新_博文表.csv')


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


def emotion_type(x):
    data = x
    new_df = data['情感类型'].value_counts()
    x_data1 = [x for x in new_df.index]
    y_data1 = [y for y in new_df.values]
    d = {}
    for x,y in zip(x_data1,y_data1):
        d[x] = y
    return d


emotion_df = df.groupby('博文发布月份').apply(emotion_type)
# 提取正面和负面数据
emotion_df = emotion_df.apply(pd.Series)  # 将字典拆分为多列
emotion_df['总数'] = emotion_df['正面'] + emotion_df['负面']  # 计算每月总数
emotion_df['正面百分比'] = (emotion_df['正面'] / emotion_df['总数']) * 100  # 计算正面百分比
emotion_df['负面百分比'] = (emotion_df['负面'] / emotion_df['总数']) * 100  # 计算负面百分比

# 绘制百分比趋势柱状图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 6))
emotion_df[['正面百分比', '负面百分比']].plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(12, 8))
plt.title('每月正负面帖子的占比趋势')
plt.xlabel('博文发布月份')
plt.ylabel('百分比 (%)')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.savefig('每月正负面帖子的占比趋势.png')
plt.show()