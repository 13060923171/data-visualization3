import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('new_data.csv')

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


def emotion():
    new_df = df['情感分值'].resample('M').mean()
    plt.figure(figsize=(20,9),dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(new_df, color='#b82410',linewidth=3)
    plt.title('情感走势')
    plt.xlabel('年份')
    plt.ylabel('数值')
    plt.grid()
    plt.savefig('情感趋势.png')
    plt.show()


def line1():
    df['transfer'] = df['transfer'].apply(main2)
    df.dropna(subset=['transfer'],axis=0,inplace=True)
    df['transfer'] = df['transfer'].astype('int')
    new_df = df['transfer'].resample('M').sum()
    plt.figure(figsize=(20, 9), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(new_df, color='#48C9B0',linewidth=3)
    plt.title('转发量走势')
    plt.xlabel('年份')
    plt.ylabel('份数')
    plt.grid()
    plt.savefig('转发量走势.png')
    plt.show()

def bar1():
    def main3(x):
        x1 = int(x)
        if x1 <= 10000:
            return '粉丝数一万以内'
        elif 10000 < x1 <= 50000:
            return '粉丝数在1万-5万区间'
        elif 50000 < x1 <= 100000:
            return '粉丝数在5万-10万区间'
        else:
            return '粉丝数大于10万'

    df['num_fans'] = df['num_fans'].apply(main3)
    new_df = df['num_fans'].value_counts()
    new_df = new_df.sort_index()
    plt.figure(figsize=(12, 16),dpi=500)
    x_data = [x for x in new_df.index]
    y_data = [int(y) for y in new_df.values]

    plt.bar(x_data,y_data, color='#EC7063')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("粉丝分布状况")
    plt.xlabel("粉丝数类别")
    plt.ylabel("数量")
    plt.xticks(rotation=65)
    plt.savefig('粉丝分布状况.png')
    plt.show()

def pie1():
    new_df = df.sort_values(by='num_fans', ascending=False)
    x_data = new_df['user_id']
    x_data = x_data[:20]

    def number(x):
        df1 = df[df['user_id'] == x]
        sum_number = df1['发帖数量'].sum()
        return sum_number
    y_data = []
    for x in x_data:
        y = number(x)
        y_data.append(y)
    x_data1 = []
    for x in range(len(x_data)):
        x = x + 1
        x1 = 'top-{}'.format(x)
        x_data1.append(x1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16,9),dpi=500)
    plt.pie(y_data, labels=x_data1, startangle=0, autopct='%1.2f%%')
    plt.title("粉丝ToP20_发帖情况")
    plt.savefig('粉丝ToP20_发帖情况.png')
    plt.show()


def line2():
    new_df = df['发帖数量'].resample('M').sum()
    df1 = pd.DataFrame()
    df1['year'] = new_df.index
    df1['values'] = new_df.values
    # 计算同比增长
    df1['growth'] = df1['values'].pct_change() * 100
    print(df1)
    # 设置图形大小
    plt.figure(figsize=(20, 9),dpi=500)

    # 创建柱状图
    plt.bar(df1['year'], df1['values'], color='lightblue', edgecolor='black', linewidth=1, width=0.5,label='月度微博数量')

    # 创建折线图
    plt.plot(df1['year'], df1['growth'], color='red', marker='o', linestyle='-', linewidth=2, markersize=8,label='月度增长率')
    # 添加数据点标签
    for i, value in enumerate(df1['growth']):
        if pd.notnull(value):
            plt.text(df1['year'][i], value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    # 添加标题和轴标签
    plt.title('月度发帖数量与增长趋势')
    plt.xlabel('Year')
    plt.ylabel('数量/增长率')
    # 显示网格线
    plt.grid(axis='y', linestyle='dashed', alpha=0.5)
    # 自动旋转 x 轴标签
    plt.xticks(rotation=45)

    # 显示图例
    plt.legend(['Growth'])
    plt.tight_layout()
    # 展示图形
    plt.savefig('月度发帖数量与增长趋势.png')
    # 展示图形

    plt.show()
    df1.to_csv('月度发帖数量与增长趋势.csv',encoding='utf-8-sig')


if __name__ == '__main__':
    # emotion()
    # line1()
    # bar1()
    pie1()
    # line2()