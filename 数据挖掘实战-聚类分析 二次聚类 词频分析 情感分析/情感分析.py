import pandas as pd
import numpy as np
from snownlp import SnowNLP

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

df = pd.read_csv('data2.csv')


# 定义情感档位划分函数
def sentiment_to_level(sentiment):
    if sentiment <= 1/6:
        return "非常负面"
    elif sentiment <= 2/6:
        return "负面"
    elif sentiment <= 3/6:
        return "轻微负面"
    elif sentiment <= 4/6:
        return "中性"
    elif sentiment <= 5/6:
        return "轻微正面"
    else:
        return "正面"


def main1(x):
    text = str(x)
    s = SnowNLP(text)
    sentiment = s.sentiments
    sentiment_level = sentiment_to_level(sentiment)
    return sentiment_level


df['情感维度'] = df['fenci'].apply(main1)
new_df = df['情感维度'].value_counts()

x_data = [x for x in new_df.index]
y_data = [y for y in new_df.values]

# 计算百分比
total = sum(y_data)
percentages = [f'{(y / total) * 100:.2f}%' for y in y_data]

# 设置绘图风格
sns.set(style="whitegrid")

# 创建柱状图
plt.figure(figsize=(10, 6))
bars = sns.barplot(x=x_data, y=y_data, palette="viridis")

# 在每个柱状图顶部显示百分比
for bar, percentage in zip(bars.patches, percentages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, percentage, ha='center', va='bottom', fontsize=12)

# 设置标题和标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False
plt.title('情感维度百分比占比')
plt.xlabel('情感维度')
plt.ylabel('数量')

# 显示图形
plt.savefig('情感维度百分比占比.png')
df.to_excel('形象情感表.xlsx',index=False)