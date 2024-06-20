import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端


def year_pie():
    def datetime2(x):
        x = str(x).split("年")
        x1 = x[0]
        return x1
    df['pubtime'] = df['pubtime'].apply(datetime2)

    new_df = df['pubtime'].value_counts()
    x_data = [x for x in new_df.index]
    for x in x_data:
        df1 = df[df['pubtime'] == x]

        new_df1 = df1['主题类型'].value_counts()

        x_data = [x for x in new_df1.index]
        y_data = [y for y in new_df1.values]

        plt.figure(figsize=(9,6),dpi=500)
        plt.rcParams['font.sans-serif'] = ['SimHei']

        plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
        plt.title('{}-主题分布'.format(x))
        plt.tight_layout()
        # 添加图例
        plt.legend(x_data, loc='lower right')
        plt.savefig('./LDA/{}-主题分布.png'.format(x))


def year_bar1():
    new_df1 = df['主题类型'].value_counts()
    x_data = [x for x in new_df1.index]

    for x in x_data:
        df1 = df[df['主题类型'] == x]
        new_df2 = df1['情感类型'].value_counts()
        new_df2 = new_df2.sort_index()
        x_data = [x for x in new_df2.index]
        y_data = [y for y in new_df2.values]

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure(figsize=(9, 6))
        # 绘制条形图
        bars = plt.bar(x_data, y_data)

        # 在每个条形上显示数值
        for bar, y in zip(bars, y_data):
            plt.text(bar.get_x() + bar.get_width() / 2, y + 0.05, '%d' % y, ha='center', va='bottom')

        # 设置标题、坐标轴标签等
        plt.title("主题{}-情感分布情况".format(x))
        plt.xlabel("情感分类")
        plt.ylabel("次数")

        # 保存图像
        plt.savefig('./LDA/主题{}-情感分布情况.png'.format(x))


def year_bar2():
    def datetime2(x):
        x = str(x).split("年")
        x1 = x[0]
        return x1
    df['pubtime'] = df['pubtime'].apply(datetime2)

    new_df = df['pubtime'].value_counts()
    x_data = [x for x in new_df.index]
    for x in x_data:
        df3 = df[df['pubtime'] == x]
        new_df1 = df3['主题类型'].value_counts()
        x_data1 = [x for x in new_df1.index]

        for x1 in x_data1:
            df1 = df3[df3['主题类型'] == x1]
            new_df2 = df1['情感类型'].value_counts()
            new_df2 = new_df2.sort_index()
            x_data2 = [x for x in new_df2.index]
            y_data2 = [y for y in new_df2.values]

            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.figure(figsize=(9, 6))
            # 绘制条形图
            bars = plt.bar(x_data2, y_data2)

            # 在每个条形上显示数值
            for bar, y in zip(bars, y_data2):
                plt.text(bar.get_x() + bar.get_width() / 2, y + 0.05, '%d' % y, ha='center', va='bottom')

            # 设置标题、坐标轴标签等
            plt.title("{}年-主题{}-情感分布情况".format(x,x1))
            plt.xlabel("情感分类")
            plt.ylabel("次数")

            # 保存图像
            plt.savefig('./LDA/{}年-主题{}-情感分布情况.png'.format(x,x1))


if __name__ == '__main__':
    df = pd.read_csv('./LDA/lda_data.csv')
    def datetime1(x):
        x = str(x).split("日")
        x1 = x[0] + '日'
        return x1
    df['pubtime'] = df['pubtime'].apply(datetime1)
    # 如果日期列是字符串，使用pd.to_datetime转换，并指定格式
    date_format = "%Y年%m月%d日"
    df.index = pd.to_datetime(df['pubtime'], format=date_format)
    # 按日期排序
    df = df.sort_index()
    year_pie()
    year_bar1()
    year_bar2()


