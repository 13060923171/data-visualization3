import pandas as pd
import numpy as np


def main1():
    df1 = pd.read_excel('comment.xlsx')
    df1['fenci'] = df1['博文发布者'] + " " + df1['评论者名称']

    df2 = df1[(df1['评论回复数量'] >= 3) & (df1['评论转发数量'] >= 3)]

    df2.to_csv('new_comment.csv',index=False,encoding='utf-8-sig')


    df3 = pd.read_csv('repost.csv')
    df4 = df3[df3['博文id'].isin(df2['博文id'].tolist())]
    df4['fenci'] = df4['博文发布者'] + " " + df4['转发者名称']
    df4 = df4.dropna(subset=['fenci'],axis=0)
    df4.to_csv('new_repost.csv',index=False,encoding='utf-8-sig')


def main2():
    df1 = pd.read_csv('new_repost.csv')
    data1 = pd.DataFrame()
    data1['fenci'] = df1['fenci']
    df2 = pd.read_csv('new_comment.csv')
    data2 = pd.DataFrame()
    data2['fenci'] = df2['fenci']
    data3 = pd.concat([data1,data2],axis=0)
    data3.to_csv('new_data.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main2()

