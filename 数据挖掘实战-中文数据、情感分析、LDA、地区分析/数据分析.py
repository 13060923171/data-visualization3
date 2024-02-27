import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pyecharts import options as opts
from pyecharts.charts import Map


def emotion_analysis():
    df = pd.read_csv('数据.csv')
    new_df = df['情感分析'].value_counts()
    # 计算每行的占比

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6), dpi=500)

    x_data = [str(x) for x in new_df.index]
    y_data = [int(x) for x in new_df.values]
    z_data = []
    for y in y_data:
        proportions = y / sum(y_data)
        proportions = str(float(round(proportions,4)) * 100) + "%"
        z_data.append(proportions)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('情感分析')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('情感分析.png')

    data = pd.DataFrame()
    data['情感分类'] = x_data
    data['数量'] = y_data
    data['占比'] = z_data
    data.to_excel('情感分析.xlsx',index=False)

    data1 = df[df['情感分析'] == '积极态度']
    data1.to_excel('积极语句.xlsx',index=False)
    data2 = df[df['情感分析'] == '消极态度']
    data2.to_excel('消极语句.xlsx', index=False)

def map_analyze():
    df = pd.read_csv('数据.csv')
    df = df.dropna(subset=['评论ip属地'],axis=0)
    new_df1 = df['评论ip属地'].value_counts()
    d1 = {}
    for x,y in zip(new_df1.index,new_df1.values):
        d1[x] = y


    ls = list(d1.items())
    ls.sort(key=lambda x:x[1],reverse=True)
    print(ls)
    data = []
    provinces = ['安徽', '澳门', '北京', '重庆', '福建', '甘肃', '广东', '广西', '贵州', '海南', '河北', '黑龙江', '河南', '湖北', '湖南', '江苏', '江西',
                 '吉林', '辽宁', '内蒙古', '宁夏', '青海', '山东', '上海', '山西', '陕西', '四川', '台湾', '天津', '西藏', '香港', '新疆', '云南', '浙江']
    x_data = []
    y_data = []
    for key,values in ls:
        if key in provinces:
            if '北京' == key or '重庆' == key or '上海' == key or '天津' == key:
                key = str(key) + '市'
            elif '广西' != key or '新疆' != key or '宁夏' != key or '西藏' != key:
                key = str(key) + '省'
            else:
                key = str(key).replace('广西','广西壮族自治区').replace('新疆','新疆维吾尔自治区').replace('宁夏','宁夏回族自治区').replace('西藏','西藏自治区')
            data.append((key,int(values)))
            x_data.append(key)
            y_data.append(values)

    df = pd.DataFrame()
    df['area'] = x_data
    df['values'] = y_data

    df.to_csv('地区分析.csv',encoding='utf-8-sig')

    c = (
        Map()
            .add("中国地图", data, "china")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
            title_opts=opts.TitleOpts(title="地域分析"),
            visualmap_opts=opts.VisualMapOpts(max_=200, is_piecewise=True),
        )
            .render("地域分析.html")
    )

    # c = (
    #     Geo()
    #         .add_schema(maptype="china")
    #         .add("中国地图", data)
    #         .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    #         .set_global_opts(
    #         title_opts=opts.TitleOpts(title="地域分析"),
    #         visualmap_opts=opts.VisualMapOpts(is_piecewise=True,max_=int(y_data[0])),
    #     )
    #         .render("地域分析.html")
    # )


def process_time():
    df = pd.read_csv('数据.csv')
    df = df.dropna(subset=['评论供应商回复时间'], axis=0)
    df['评论时间'] = pd.to_datetime(df['评论时间'])
    df['评论供应商回复时间'] = pd.to_datetime(df['评论供应商回复时间'])

    # 计算时间差
    df['time_difference(单位：天)'] =  df['评论供应商回复时间'] - df['评论时间']

    df.to_excel('时间相差数据.xlsx',index=False)

def myd_data(area):
    df = pd.read_csv('数据.csv')
    df1 = df[df['地区分布'] == area]
    new_df = df1['产品名称'].value_counts()
    number_name = len(new_df)
    df2 = df1.drop_duplicates(subset=['产品名称'], keep='first')
    mean_proice =int(df2['票价'].mean())
    people_number = df2['出游人数'].sum()
    sum_myd = round(df2['产品满意度'].mean(),2)
    new_df2 = df1['情感分析'].value_counts()
    new_df2 = new_df2.sort_index()
    x_data = [str(x) for x in new_df2.index]
    y_data = [int(x) for x in new_df2.values]
    z_data = []
    for y in y_data:
        proportions = y / sum(y_data)
        proportions = str(float(round(proportions, 4)) * 100) + "%"
        z_data.append(proportions)

    new_df3 = pd.DataFrame()
    new_df3['地区分布'] = [area]
    new_df3['产品线路'] = [number_name]
    new_df3['出游总人数'] = [people_number]
    new_df3['平均单价'] = [mean_proice]
    new_df3['整体好评率'] = [sum_myd]
    new_df3['好评占比'] = [z_data[1]]
    new_df3['差评占比'] = [z_data[0]]

    new_df3.to_csv('区域分析.csv',encoding='utf-8-sig',index=False,header=False,mode='a+')

def travel_type(area):
    df = pd.read_csv('数据.csv')
    df1 = df[df['地区分布'] == area]
    new_df2 = df1['评论出行类型'].value_counts()
    new_df2 = new_df2.sort_index()
    x_data = [str(x) for x in new_df2.index]
    y_data = [int(x) for x in new_df2.values]
    z_data = {}
    for x,y in zip(x_data,y_data):
        proportions = y / sum(y_data)
        proportions = str(float(round(proportions, 4)) * 100) + "%"
        z_data[x] = proportions

    new_df3 = pd.DataFrame()
    new_df3['地区分布'] = [area]
    new_df3['其他出游'] = [z_data['其他出游']]
    new_df3['单独旅行'] = [z_data['单独旅行']]
    new_df3['家庭亲子'] = [z_data['家庭亲子']]
    new_df3['情侣夫妻'] = [z_data['情侣夫妻']]
    new_df3['朋友出游'] = [z_data['朋友出游']]

    new_df3.to_csv('出行类型分析.csv', encoding='utf-8-sig', index=False, header=False, mode='a+')

def score(area):
    df = pd.read_csv('数据.csv')
    df1 = df[df['地区分布'] == area]
    new_df2 = df1['评分'].value_counts()
    new_df2 = new_df2.sort_index()
    new_df2.to_excel('{}-评分统计.xlsx'.format(area),index=False)
    x_data = [str(x) for x in new_df2.index]
    y_data = [int(x) for x in new_df2.values]

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 9), dpi=500)
    # 创建画布和子图
    fig, ax1 = plt.subplots()

    # 绘制折线图
    ax1.plot(x_data, y_data, 'b-')
    ax1.set_ylabel('数量',color='b')

    # 创建第二个 y 轴
    ax2 = ax1.twinx()

    # 绘制柱状图
    ax2.bar(x_data, y_data, alpha=0.5, color='r')
    ax2.set_ylabel(' ',color='r')
    ax2.set_xlabel('评分')
    # 设置 x 轴和标题
    plt.xticks(x_data, [f'{i}' for i in x_data], rotation=45)
    plt.title('{}-评分趋势'.format(area))
    plt.savefig('{}-评分趋势.png'.format(area))

if __name__ == '__main__':
    # emotion_analysis()
    map_analyze()
    # process_time()
    # df = pd.DataFrame()
    # new_df3 = pd.DataFrame()
    # new_df3['地区分布'] = ['地区分布']
    # new_df3['产品线路'] = ['产品线路']
    # new_df3['出游总人数'] = ['出游总人数']
    # new_df3['平均单价'] = ['平均单价']
    # new_df3['整体好评率'] = ['整体好评率']
    # new_df3['好评占比'] =['好评占比']
    # new_df3['差评占比'] = ['差评占比']
    # new_df3.to_csv('区域分析.csv', encoding='utf-8-sig', index=False, header=False, mode='w')
    # new_df4 = pd.DataFrame()
    # new_df4['地区分布'] = ['地区分布']
    # new_df4['其他出游'] = ['其他出游']
    # new_df4['单独旅行'] = ['单独旅行']
    # new_df4['家庭亲子'] = ['家庭亲子']
    # new_df4['情侣夫妻'] = ['情侣夫妻']
    # new_df4['朋友出游'] = ['朋友出游']
    # new_df4.to_csv('出行类型分析.csv', encoding='utf-8-sig', index=False, header=False, mode='w')
    # list_area = ['东南亚','欧美','日韩']
    # for l in list_area:
    #     myd_data(l)
    #     travel_type(l)
    #     score(l)
