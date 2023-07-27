import pandas as pd
import matplotlib.pyplot as plt


def demo1():
    df = pd.read_csv('./result1/new_data.csv')
    def main1(x):
        x1 = str(x).split(" ")
        x2 = x1[0] + " "+ x1[1] + " "+ x1[2] + " " + x1[-1]
        return x2
    df['发帖数量'] = 1
    def emtion_number(x):
        data = x
        new_data = data['comp_score'].value_counts()
        x_data = [x for x in new_data.index]
        y_data = [y for y in new_data.values]
        d = {}
        for x,y in zip(x_data,y_data):
            d[x] = y
        return d

    def topic_number(x):
        data = x
        new_data = data['主题类型'].value_counts()
        x_data = [x for x in new_data.index]
        y_data = [y for y in new_data.values]
        d = {}
        for x,y in zip(x_data,y_data):
            d['Topic_{}'.format(x)] = y
        return d

    df['发文时间'] = df['发文时间'].apply(main1)
    df['发文时间'] = pd.to_datetime(df['发文时间'])
    data1 = df.groupby('发文时间').apply(emtion_number)
    data2 = df.groupby('发文时间').apply(topic_number)

    new_df = df.groupby('发文时间').agg('mean')
    new_df1 = df.groupby('发文时间').agg('sum')
    df1 = pd.DataFrame()
    df1['发帖数量'] = new_df1['发帖数量']
    df1['情感类别'] = data1.values
    df1['主题数量'] = data2.values
    df1['转发'] = new_df1['转发']
    df1['点赞'] = new_df1['点赞']
    df1['评论'] = new_df1['评论']
    df1['熵值'] = new_df['entropy_values']

    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(list(df1.index),df1['转发'], color='#A93226',linewidth=3)
    plt.title('retweet trend')
    plt.xlabel('DAY')
    plt.ylabel('value')
    plt.grid()
    plt.savefig('./result1/retweet trend.png')
    plt.show()


    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(list(df1.index),df1['点赞'], color='#3498DB',linewidth=3)
    plt.title('like trend')
    plt.xlabel('DAY')
    plt.ylabel('value')
    plt.grid()
    plt.savefig('./result1/like trend.png')
    plt.show()

    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(list(df1.index),df1['评论'], color='#F39C12',linewidth=3)
    plt.title('comment trend')
    plt.xlabel('DAY')
    plt.ylabel('value')
    plt.grid()
    plt.savefig('./result1/comment trend.png')
    plt.show()


    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(list(df1.index),df1['熵值'], color='#2E4053',linewidth=3)
    plt.title('entropy trend')
    plt.xlabel('DAY')
    plt.ylabel('value')
    plt.grid()
    plt.savefig('./result1/entropy trend.png')
    plt.show()

    df1.to_csv('./result1/时间数据.csv',encoding='utf-8-sig')


def demo2():
    df = pd.read_csv('./result2/new_data.csv')
    def main1(x):
        x1 = str(x).split(" ")
        x2 = x1[0]
        return x2

    df['发帖数量'] = 1

    def emtion_number(x):
        data = x
        new_data = data['comp_score'].value_counts()
        x_data = [x for x in new_data.index]
        y_data = [y for y in new_data.values]
        d = {}
        for x, y in zip(x_data, y_data):
            d[x] = y
        return d

    def topic_number(x):
        data = x
        new_data = data['主题类型'].value_counts()
        x_data = [x for x in new_data.index]
        y_data = [y for y in new_data.values]
        d = {}
        for x, y in zip(x_data, y_data):
            d['Topic_{}'.format(x)] = y
        return d

    df['发文时间'] = df['发文时间'].apply(main1)
    df['发文时间'] = pd.to_datetime(df['发文时间'])
    data1 = df.groupby('发文时间').apply(emtion_number)
    data2 = df.groupby('发文时间').apply(topic_number)

    new_df = df.groupby('发文时间').agg('mean')
    new_df1 = df.groupby('发文时间').agg('sum')
    df1 = pd.DataFrame()
    df1['发帖数量'] = new_df1['发帖数量']
    df1['情感类别'] = data1.values
    df1['主题数量'] = data2.values
    df1['转发'] = new_df1['转发']
    df1['点赞'] = new_df1['点赞']
    df1['评论'] = new_df1['评论']
    df1['熵值'] = new_df['entropy_values']

    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(list(df1.index),df1['转发'], color='#A93226',linewidth=3)
    plt.title('retweet trend')
    plt.xlabel('DAY')
    plt.ylabel('value')
    plt.grid()
    plt.savefig('./result2/retweet trend.png')
    plt.show()


    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(list(df1.index),df1['点赞'], color='#3498DB',linewidth=3)
    plt.title('like trend')
    plt.xlabel('DAY')
    plt.ylabel('value')
    plt.grid()
    plt.savefig('./result2/like trend.png')
    plt.show()

    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(list(df1.index),df1['评论'], color='#F39C12',linewidth=3)
    plt.title('comment trend')
    plt.xlabel('DAY')
    plt.ylabel('value')
    plt.grid()
    plt.savefig('./result2/comment trend.png')
    plt.show()


    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(list(df1.index),df1['熵值'], color='#2E4053',linewidth=3)
    plt.title('entropy trend')
    plt.xlabel('DAY')
    plt.ylabel('value')
    plt.grid()
    plt.savefig('./result2/entropy trend.png')
    plt.show()

    df1.to_csv('./result2/时间数据.csv',encoding='utf-8-sig')


if __name__ == '__main__':
    demo1()
    demo2()