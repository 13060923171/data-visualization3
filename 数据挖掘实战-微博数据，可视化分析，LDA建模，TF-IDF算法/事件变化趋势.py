import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_excel('./data/微博数据.xlsx')
df2 = pd.read_excel('./data/微博数据1.xlsx')
df3 = pd.read_excel('./data/微博数据2.xlsx')
df4 = pd.read_excel('./data/微博数据3.xlsx')
df = pd.concat([df1,df2,df3,df4],axis=0)

def main1(x):
    x1 = str(x)
    x1 = x1.split('日')
    x1 = x1[0]
    if '年' not in x1:
        x1 = '2023年' + str(x1)
        x1 = x1.replace('年','-').replace('月','-')
    else:
        x1 = x1.replace('年', '-').replace('月', '-')
    return x1


def main2(x):
    x1 = str(x)
    x1 = x1.replace('转发','').strip(' ')
    if len(x1)!= 0:
        return x1
    else:
        return np.NaN


df['pubtime'] = df['pubtime'].apply(main1)
df['pubtime'] = pd.to_datetime(df['pubtime'])
df['发帖数量'] = 1
df.index = df['pubtime']

#舆情爆发阶段
df1 = df['2021-07-20':'2021-08-03']
#舆情平淡阶段
df2 = df['2022-07-24':'2022-08-01']


# def emotion(df,name):
#     new_df = df['情感分值'].resample('W').mean()
#     plt.figure(figsize=(20,9),dpi=500)
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.plot(new_df, color='#A93226',linewidth=3)
#     plt.title('{}_emotional trend'.format(name))
#     plt.xlabel('week')
#     plt.ylabel('value')
#     plt.grid()
#     plt.savefig('{}_emotional trend.png'.format(name))
#     plt.show()


def line1(df,name):
    df['transfer'] = df['transfer'].apply(main2)
    df.dropna(subset=['transfer'],axis=0,inplace=True)
    df['transfer'] = df['transfer'].astype('int')
    new_df = df['transfer'].resample('D').sum()
    plt.figure(figsize=(20, 9), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(new_df, color='#F39C12',linewidth=3)
    plt.title('{}_Forwarding trend_late'.format(name))
    plt.xlabel('week')
    plt.ylabel('value')
    plt.grid()
    plt.savefig('./data/{}_Forwarding trend_late.png'.format(name))
    plt.show()


def line2(df,name):
    new_df = df['发帖数量'].resample('D').sum()
    plt.figure(figsize=(20, 9), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(new_df, color='#3498DB',linewidth=3)
    plt.title('{}_Post trend_late'.format(name))
    plt.xlabel('week')
    plt.ylabel('value')
    plt.grid()
    plt.savefig('./data/{}_Post trend_late.png'.format(name))
    plt.show()


if __name__ == '__main__':
    data = [df1,df2]
    name1 = ['Carbon emissions trading market opens','european heat wave']
    for d,n in zip(data,name1):
        # emotion(d,n)
        line1(d,n)
        line2(d,n)