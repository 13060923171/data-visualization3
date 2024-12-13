import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


def process1(x):
    df = pd.read_excel(f'{x}.xlsx')
    df1 = df[df['类型'] == '博文']
    sum_number = len(df1)
    pos_number1 = df1['正面分值'].mean()
    neg_number1 = df1['负面分值'].mean()
    return sum_number,pos_number1,neg_number1

def process2(x):
    df = pd.read_excel(f'{x}.xlsx')
    df1 = df[df['类型'] == '评论']
    sum_number = len(df1)
    pos_number1 = df1['正面分值'].mean()
    neg_number1 = df1['负面分值'].mean()
    return sum_number,pos_number1,neg_number1


def bar1():
    list_name3 = ['1-17博文&评论', '18博文&评论', '19-24博文&评论', '25-31博文&评论']
    list_sum = []
    list_pos = []
    list_neg = []
    for l in list_name3:
        sum_number,pos_number1,neg_number1 = process1(l)
        list_sum.append(sum_number)
        list_pos.append(pos_number1)
        list_neg.append(neg_number1)

    # 创建 DataFrame
    emotion_df = pd.DataFrame()
    emotion_df['时间类型'] = ['1-17', '18', '19-24', '25-31']
    emotion_df['正面百分比'] = list_pos
    emotion_df['负面百分比'] = list_neg

    # 绘制曲线图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 9), dpi=500)

    # 绘制正面情感曲线
    plt.plot(emotion_df['时间类型'], emotion_df['正面百分比'], marker='o', label='正面均值', color='blue')

    # 显示每个点的数值
    for x, y in zip(emotion_df['时间类型'], emotion_df['正面百分比']):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', color='black')


    plt.title('博文帖子情感趋势变化')
    plt.xlabel('时间')
    plt.ylabel('情感均值')
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    plt.savefig('博文帖子情感趋势变化.png', dpi=500)
    plt.show()


def bar2():
    list_name3 = ['1-17博文&评论', '18博文&评论', '19-24博文&评论', '25-31博文&评论']
    list_sum = []
    list_pos = []
    list_neg = []
    for l in list_name3:
        sum_number, pos_number1, neg_number1 = process2(l)
        list_sum.append(sum_number)
        list_pos.append(pos_number1)
        list_neg.append(neg_number1)

    # 创建 DataFrame
    emotion_df = pd.DataFrame()
    emotion_df['时间类型'] = ['1-17', '18', '19-24', '25-31']
    emotion_df['正面百分比'] = list_pos
    emotion_df['负面百分比'] = list_neg

    # 绘制曲线图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 9), dpi=500)

    # 绘制正面情感曲线
    plt.plot(emotion_df['时间类型'], emotion_df['正面百分比'], marker='o', label='正面均值', color='blue')

    # 显示每个点的数值
    for x, y in zip(emotion_df['时间类型'], emotion_df['正面百分比']):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', color='black')

    plt.title('评论情感趋势变化')
    plt.xlabel('时间')
    plt.ylabel('情感均值')
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    plt.savefig('评论情感趋势变化.png', dpi=500)
    plt.show()


def emotion_pie(x1,x2):
    df = pd.read_excel(f'{x1}.xlsx')
    df1 = df[df['类型'] == '博文']
    new_df = df1['标签'].value_counts()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(9, 6), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title(f'{x2}博文-情感占比分布')
    plt.tight_layout()
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.savefig(f'{x2}博文-情感占比分布.png')


def emotion_pie2(x1,x2):
    df = pd.read_excel(f'{x1}.xlsx')
    df1 = df[df['类型'] == '评论']
    new_df = df1['标签'].value_counts()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(9, 6), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title(f'{x2}评论-情感占比分布')
    plt.tight_layout()
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.savefig(f'{x2}评论-情感占比分布.png')


if __name__ == '__main__':
    bar1()
    bar2()
    list_name3 = ['1-17博文&评论', '18博文&评论', '19-24博文&评论', '25-31博文&评论']
    list_name1 = ['1-17', '18', '19-24', '25-31']
    for l1,l2 in zip(list_name3,list_name1):
        emotion_pie(l1,l2)
        emotion_pie2(l1,l2)




