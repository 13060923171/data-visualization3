import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


def main1():
    df = pd.read_csv('new_data.csv')
    df1 = df[df['视频vid'] == 'BV1x54y1e7zf']
    df1 = df1.sort_values(by='创建时间', ascending=True)
    def time_process(x):
        x1 = str(x).split("-")
        x1 = x1[0]
        return x1

    def month_pd(df):
        new_df = df['情感类别'].value_counts()
        new_df = new_df.sort_index()
        type1 = [x for x in new_df.index]
        number1 = [x for x in new_df.values]
        return type1,number1
    df1['年份'] = df1['创建时间'].apply(time_process)
    df2 = df1[df1['评论楼层'] == '一级评论']
    df3 = df2.groupby('年份').apply(month_pd)
    # 初始化一个空的列表，用于存储所有的年份、类别和数量
    data = []
    # 遍历 df3，提取每个年份的数据
    for year, (types, counts) in df3.items():
        for t, c in zip(types, counts):
            data.append([year, t, c])

    # 创建一个新的 DataFrame 来存储整理后的数据
    new_df1 = pd.DataFrame(data, columns=['一级评论_年份', '一级评论_类别', '一级评论_数量'])

    df4 = df1[df1['评论楼层'] == '二级评论']
    df5 = df4.groupby('年份').apply(month_pd)

    data1 = []
    for year, (types, counts) in df5.items():
        for t, c in zip(types, counts):
            data1.append([year, t, c])

    # 创建一个新的 DataFrame 来存储整理后的数据
    new_df2 = pd.DataFrame(data1, columns=['二级评论_年份', '二级评论_类别', '二级评论_数量'])
    new_df = pd.concat([new_df1,new_df2],axis=1)
    new_df.to_excel('BV1x54y1e7zf_评论数量统计.xlsx',index=False)


def main2():
    df = pd.read_csv('new_data.csv')
    df1 = df[(df['视频vid'] == 'BV1GE4m1d7KK') | (df['视频vid'] == 'BV1Ti421a7dv')]
    df1 = df1.sort_values(by='创建时间', ascending=True)

    def time_process(x):
        x1 = str(x).split(" ")
        x1 = x1[0]
        return x1

    def month_pd(df):
        new_df = df['情感类别'].value_counts()
        new_df = new_df.sort_index()
        type1 = [x for x in new_df.index]
        number1 = [x for x in new_df.values]
        return type1, number1

    df1['日'] = df1['创建时间'].apply(time_process)

    df2 = df1[df1['评论楼层'] == '一级评论']
    df3 = df2.groupby('日').apply(month_pd)
    # 初始化一个空的列表，用于存储所有的年份、类别和数量
    data = []
    # 遍历 df3，提取每个年份的数据
    for year, (types, counts) in df3.items():
        for t, c in zip(types, counts):
            data.append([year, t, c])

    # 创建一个新的 DataFrame 来存储整理后的数据
    new_df1 = pd.DataFrame(data, columns=['一级评论_日', '一级评论_类别', '一级评论_数量'])

    df4 = df1[df1['评论楼层'] == '二级评论']
    df5 = df4.groupby('日').apply(month_pd)

    data1 = []
    for year, (types, counts) in df5.items():
        for t, c in zip(types, counts):
            data1.append([year, t, c])

    # 创建一个新的 DataFrame 来存储整理后的数据
    new_df2 = pd.DataFrame(data1, columns=['二级评论_日', '二级评论_类别', '二级评论_数量'])
    new_df = pd.concat([new_df1, new_df2], axis=1)
    new_df.to_excel('BV1GE4m1d7KK_BV1Ti421a7dv_评论数量统计1.xlsx', index=False)


def main3():
    df = pd.read_csv('new_data.csv')
    df1 = df[(df['视频vid'] == 'BV17T421z7JA') | (df['视频vid'] == 'BV1CZ421T7kD')]
    df1 = df1.sort_values(by='创建时间', ascending=True)

    def time_process(x):
        x1 = str(x).split(" ")
        x1 = x1[0]
        return x1

    def month_pd(df):
        new_df = df['情感类别'].value_counts()
        new_df = new_df.sort_index()
        type1 = [x for x in new_df.index]
        number1 = [x for x in new_df.values]
        return type1, number1

    df1['日'] = df1['创建时间'].apply(time_process)

    df2 = df1[df1['评论楼层'] == '一级评论']
    df3 = df2.groupby('日').apply(month_pd)
    # 初始化一个空的列表，用于存储所有的年份、类别和数量
    data = []
    # 遍历 df3，提取每个年份的数据
    for year, (types, counts) in df3.items():
        for t, c in zip(types, counts):
            data.append([year, t, c])

    # 创建一个新的 DataFrame 来存储整理后的数据
    new_df1 = pd.DataFrame(data, columns=['一级评论', '一级评论_类别', '一级评论_数量'])

    df4 = df1[df1['评论楼层'] == '二级评论']
    df5 = df4.groupby('日').apply(month_pd)

    data1 = []
    for year, (types, counts) in df5.items():
        for t, c in zip(types, counts):
            data1.append([year, t, c])

    # 创建一个新的 DataFrame 来存储整理后的数据
    new_df2 = pd.DataFrame(data1, columns=['二级评论', '二级评论_类别', '二级评论_数量'])
    new_df = pd.concat([new_df1, new_df2], axis=1)
    new_df.to_excel('BV17T421z7JA_BV1CZ421T7kD_评论数量统计.xlsx', index=False)


if __name__ == '__main__':
    # main1()
    # main2()
    main3()