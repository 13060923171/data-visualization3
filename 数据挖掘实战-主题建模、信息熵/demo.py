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
    df10 = pd.read_csv('./result2/new_data9.12.csv')
    # new_df = df1['企业'].value_counts()
    # x_data = [x for x in new_df.index]
    for x in ['a','u','t']:
        df = df10[df10['企业'] == x]
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
        pos_list = []
        neu_list = []
        neg_list = []
        dic = data1.values
        for d in dic:
            try:
                pos_list.append(d['pos'])
            except:
                pos_list.append(0)
            try:
                neu_list.append(d['neu'])
            except:
                neu_list.append(0)
            try:
                neg_list.append(d['neg'])
            except:
                neg_list.append(0)


        dic2 = data2.values
        Topic_0_list = []
        Topic_1_list = []
        Topic_2_list = []
        Topic_3_list = []
        Topic_4_list = []

        for d in dic2:
            try:
                Topic_0_list.append(d['Topic_0'])
            except:
                Topic_0_list.append(0)
            try:
                Topic_1_list.append(d['Topic_1'])
            except:
                Topic_1_list.append(0)
            try:
                Topic_2_list.append(d['Topic_2'])
            except:
                Topic_2_list.append(0)
            try:
                Topic_3_list.append(d['Topic_3'])
            except:
                Topic_3_list.append(0)
            try:
                Topic_4_list.append(d['Topic_4'])
            except:
                Topic_4_list.append(0)

        new_df = df.groupby('发文时间').agg('mean')
        new_df1 = df.groupby('发文时间').agg('sum')
        df1 = pd.DataFrame()
        df1['发帖数量'] = new_df1['发帖数量']
        # df1['情感类别'] = data1.values
        # df1['主题数量'] = data2.values
        df1['pos'] = pos_list
        df1['neu'] = neu_list
        df1['neg'] = neg_list
        df1['Topic_0'] = Topic_0_list
        df1['Topic_1'] = Topic_1_list
        df1['Topic_2'] = Topic_2_list
        df1['Topic_3'] = Topic_3_list
        df1['Topic_4'] = Topic_4_list
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
        plt.savefig('./result2/{}_retweet trend.png'.format(x))
        plt.show()


        plt.figure(figsize=(20,9),dpi=500)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(list(df1.index),df1['点赞'], color='#3498DB',linewidth=3)
        plt.title('like trend')
        plt.xlabel('DAY')
        plt.ylabel('value')
        plt.grid()
        plt.savefig('./result2/{}_like trend.png'.format(x))
        plt.show()

        plt.figure(figsize=(20,9),dpi=500)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(list(df1.index),df1['评论'], color='#F39C12',linewidth=3)
        plt.title('comment trend')
        plt.xlabel('DAY')
        plt.ylabel('value')
        plt.grid()
        plt.savefig('./result2/{}_comment trend.png'.format(x))
        plt.show()


        plt.figure(figsize=(20,9),dpi=500)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(list(df1.index),df1['熵值'], color='#2E4053',linewidth=3)
        plt.title('entropy trend')
        plt.xlabel('DAY')
        plt.ylabel('value')
        plt.grid()
        plt.savefig('./result2/{}_entropy trend.png'.format(x))
        plt.show()

        df1.to_csv('./result2/{}_时间数据.csv'.format(x),encoding='utf-8-sig')


if __name__ == '__main__':
    # demo1()
    demo2()