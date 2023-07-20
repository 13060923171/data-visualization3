import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


#绘制主题强度饼图
def plt_pie(time_name):
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    df = pd.read_csv('./{}/特征词.csv'.format(time_name))
    x_data = list(df['所属主题'])
    y_data = list(df['文章数量'])
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('theme strength')
    plt.tight_layout()
    plt.savefig('./{}/theme strength.png'.format(time_name))


def emtion_trend(df1):
    data = df1
    new_df = data['情感类型'].value_counts()
    new_df = new_df.sort_index()

    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    return x_data,y_data


if __name__ == '__main__':
    df1 = pd.read_csv('new_data.csv')
    df2 = pd.DataFrame()
    df2['time'] = df1['createdAt']
    df2['content'] = df1['message']
    df2['clearn_comment'] = df1['clearn_comment']
    df2['情感类型'] = df1['情感类型']
    df2['情感得分'] = df1['情感得分']

    df3 = pd.read_csv('new_data1.csv')
    df4 = pd.DataFrame()
    df4['time'] = df3['created_at']
    df4['content'] = df3['text']
    df4['clearn_comment'] = df3['clearn_comment']
    df4['情感类型'] = df3['情感类型']
    df4['情感得分'] = df3['情感得分']
    data = pd.concat([df2, df4], axis=0)

    def time_process(x):
        x1 = str(x).split(" ")
        x2 = x1[0] + " " + x1[1] + " " + x1[-1]
        return x2

    data['time'] = data['time'].apply(time_process)
    data['time'] = pd.to_datetime(data['time'])
    data.index = data['time']
    new_df = data['time'].value_counts()
    new_df = new_df.sort_index()
    x_data = [str(x) for x in new_df.index]

    x_data1 = []
    x_data2 = []
    y_data1 = []
    for x in x_data:
        x = str(x).split(" ")
        x = x[0]
        plt_pie(x)
        x_data1.append(x)
        data1 = data[x:x]
        x_data, y_data = emtion_trend(data1)
        x_data2.append(x_data)
        y_data1.append(y_data)

    NEGATIVE = [y_data1[0][0],y_data1[1][0],y_data1[2][0]]
    NEUTRAL = [y_data1[0][1],y_data1[1][1],y_data1[2][1]]
    POSITIVE = [y_data1[0][2],y_data1[1][2],y_data1[2][2]]
    # 创建子图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots()
    # 设置柱状图的宽度
    bar_width = 0.25
    # 设置位置
    x1 = np.arange(len(x_data1))
    x2 = x1 + bar_width
    x3 = x1 + 2 * bar_width

    ax.bar(x1, POSITIVE,width=bar_width,color='#E74C3C',label='POSITIVE')
    ax.bar(x2, NEUTRAL,width=bar_width, color='#F4D03F',label='NEUTRAL')
    ax.bar(x3, NEGATIVE,width=bar_width,color='#2C3E50',label='NEGATIVE')

    # 设置x轴标签和标题
    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.set_title('emotional_trend')

    # 设置x轴刻度标签
    ax.set_xticks(x1)
    ax.set_xticklabels(x_data1)

    # 添加图例
    ax.legend()

    # 展示图表
    plt.savefig('emotional_trend.png')
    plt.show()

    df = pd.read_csv('time_lda.csv')
    x_data3 = [x for x in df['日期']]
    y_data3 = [y for y in df['最优主题数']]

    plt.plot(x_data3, y_data3, color='#b82410')
    plt.legend()
    plt.title('Topic Time Evolution')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.grid()
    plt.savefig('Topic_Time_Evolution.png')
    plt.show()
