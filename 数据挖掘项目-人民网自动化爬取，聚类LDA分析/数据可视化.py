import pandas as pd
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.globals import ThemeType

def plt_pie():
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    df = pd.read_csv('./data/提问内容-特征词.csv')
    x_data = list(df['所属主题'])
    y_data = list(df['文章数量'])
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('提问内容-主题强度')
    plt.tight_layout()
    plt.savefig('./data/提问内容-主题强度.png')


def plt_pie1():
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    df1 = pd.read_csv('./data/官方答复-特征词.csv')
    x_data = list(df1['所属主题'])
    y_data = list(df1['文章数量'])
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('官方答复-主题强度')
    plt.tight_layout()
    plt.savefig('./data/官方答复-主题强度.png')


def month_number():
    df = pd.read_csv('all_data.csv',parse_dates=['提问时间'], index_col="提问时间")
    df5 = pd.read_csv('all_data.csv', parse_dates=['答复时间'], index_col="答复时间")
    df = df.dropna(subset=['提问内容'], axis=0)
    df5 = df5.dropna(subset=['官方答复'], axis=0)
    df['发布数据'] = 1
    df5['发布数据'] = 1

    df1 = df['发布数据'].resample('M').sum()
    x_data = [str(x).split(" ")[0] for x in df1.index]
    y_data = [int(y) for y in df1.values]

    df3 = df5['发布数据'].resample('M').sum()
    x_data1 = [str(x).split(" ")[0] for x in df3.index]
    y_data1 = [int(y) for y in df3.values]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(20,9),dpi=300)
    plt.bar(x_data,y_data,color='#1ABC9C')
    plt.title('提问内容-发帖分布趋势')
    plt.xlabel('月份')
    plt.ylabel('数量')
    plt.savefig('./data/提问内容-发帖分布趋势.png')
    plt.show()

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(20, 9), dpi=300)
    plt.bar(x_data1, y_data1,color='#3498DB')
    plt.title('官方答复-发帖分布趋势')
    plt.xlabel('月份')
    plt.ylabel('数量')
    plt.savefig('./data/官方答复-发帖分布趋势.png')
    plt.show()

    data = pd.DataFrame()
    data['官方-时间'] = x_data1
    data['官方-数量'] = y_data1
    data['提问-时间'] = x_data
    data['提问-数量'] = y_data
    data.to_csv('./data/发文频率.csv',encoding='utf-8-sig')


def emotion_bar():
    df = pd.read_csv('./data/nlp_all_data.csv',parse_dates=['提问时间'], index_col="提问时间")

    def main(x):
        x1 = str(x).split('：')
        x1 = x1[-1]
        if '省' in x1:
            x2 = str(x1).split('省')
            x2 = str(x2[0]) + "省"
            return x2
        elif '市' in x1 and '自治区' not in x1:
            x2 = str(x1).split('市')
            x2 = str(x2[0]) + "市"
            return x2
        elif '自治区' in x1 and '市' in x1:
            x2 = str(x1).split('自治区')
            x2 = str(x2[0]) + "自治区"
            return x2
    df['地方1'] = df['地方'].apply(main)

    def main1(x):
        x1 = float(x)
        if x1 >= 0.4:
            return 1
        else:
            return 0

    def main2(x):
        x1 = float(x)
        if x1 > 0.4:
            return 1
        else:
            return 0
    df['非负'] = df['提问内容_positive_probs'].apply(main1)
    df['负面'] = df['提问内容_negative_probs'].apply(main2)
    df1 = df.groupby('地方1').agg('sum')

    x_data = [x for x in df1.index]
    y_data = [int(y) for y in df1['非负']]
    z_data = [int(z) for z in df1['负面']]

    c = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="800px",theme=ThemeType.ESSOS))
            .add_xaxis(x_data)
            .add_yaxis("非负", y_data, stack="stack1", category_gap="50%")
            .add_yaxis("负面", z_data, stack="stack1", category_gap="50%")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=True,position="right",))
            .set_global_opts(title_opts=opts.TitleOpts(title="提问内容-各个地方情感倾向分布状况"),
                             xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),)
            .render("./data/提问内容-各个地方情感倾向分布状况.html")
    )


def emotion_bar1():
    df = pd.read_csv('./data/nlp_all_data.csv',parse_dates=['答复时间'], index_col="答复时间")

    def main(x):
        x1 = str(x).split('：')
        x1 = x1[-1]
        if '省' in x1:
            x2 = str(x1).split('省')
            x2 = str(x2[0]) + "省"
            return x2
        elif '市' in x1 and '自治区' not in x1:
            x2 = str(x1).split('市')
            x2 = str(x2[0]) + "市"
            return x2
        elif '自治区' in x1 and '市' in x1:
            x2 = str(x1).split('自治区')
            x2 = str(x2[0]) + "自治区"
            return x2
    df['地方1'] = df['地方'].apply(main)

    def main1(x):
        x1 = float(x)
        if x1 >= 0.4:
            return 1
        else:
            return 0

    def main2(x):
        x1 = float(x)
        if x1 > 0.4:
            return 1
        else:
            return 0
    df['非负'] = df['官方答复_positive_probs'].apply(main1)
    df['负面'] = df['官方答复_negative_probs'].apply(main2)
    df1 = df.groupby('地方1').agg('sum')

    x_data = [x for x in df1.index]
    y_data = [int(y) for y in df1['非负']]
    z_data = [int(z) for z in df1['负面']]

    c = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="800px",theme=ThemeType.ESSOS))
            .add_xaxis(x_data)
            .add_yaxis("非负", y_data, stack="stack1", category_gap="50%")
            .add_yaxis("负面", z_data, stack="stack1", category_gap="50%")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=True,position="right",))
            .set_global_opts(title_opts=opts.TitleOpts(title="官方答复-各个地方情感倾向分布状况"),
                             xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),)
            .render("./data/官方答复-各个地方情感倾向分布状况.html")
    )


def lda_bar():
    left = pd.read_csv('./data/nlp_all_data.csv')
    right = pd.read_csv('./data/提问内容-all_data.csv')

    right['提问内容'] = right['内容']

    df = pd.merge(left, right, on='提问内容')

    def main(x):
        x1 = str(x).split('：')
        x1 = x1[-1]
        if '省' in x1:
            x2 = str(x1).split('省')
            x2 = str(x2[0]) + "省"
            return x2
        elif '市' in x1 and '自治区' not in x1:
            x2 = str(x1).split('市')
            x2 = str(x2[0]) + "市"
            return x2
        elif '自治区' in x1 and '市' in x1:
            x2 = str(x1).split('自治区')
            x2 = str(x2[0]) + "自治区"
            return x2
    df['地方1'] = df['地方'].apply(main)

    def main1(x):
        x1 = float(x)
        if x1 >= 0.4:
            return 1
        else:
            return 0

    def main2(x):
        x1 = float(x)
        if x1 > 0.4:
            return 1
        else:
            return 0
    df['非负'] = df['提问内容_positive_probs'].apply(main1)
    df['负面'] = df['提问内容_negative_probs'].apply(main2)
    df1 = df.groupby('主题类型').agg('sum')

    x_data = [x for x in df1.index]
    y_data = [int(y) for y in df1['非负']]
    z_data = [int(z) for z in df1['负面']]

    c = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="800px",theme=ThemeType.ESSOS))
            .add_xaxis(x_data)
            .add_yaxis("非负", y_data, stack="stack1", category_gap="50%")
            .add_yaxis("负面", z_data, stack="stack1", category_gap="50%")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=True,position="right",))
            .set_global_opts(title_opts=opts.TitleOpts(title="提问内容-各个主题情感倾向分布状况"),
                             xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),)
            .render("./data/提问内容-各个主题情感倾向分布状况.html")
    )

def lda_bar1():
    left = pd.read_csv('./data/nlp_all_data.csv')
    right = pd.read_csv('./data/官方答复-all_data.csv')

    right['官方答复'] = right['内容']

    df = pd.merge(left, right, on='官方答复')

    def main(x):
        x1 = str(x).split('：')
        x1 = x1[-1]
        if '省' in x1:
            x2 = str(x1).split('省')
            x2 = str(x2[0]) + "省"
            return x2
        elif '市' in x1 and '自治区' not in x1:
            x2 = str(x1).split('市')
            x2 = str(x2[0]) + "市"
            return x2
        elif '自治区' in x1 and '市' in x1:
            x2 = str(x1).split('自治区')
            x2 = str(x2[0]) + "自治区"
            return x2
    df['地方1'] = df['地方'].apply(main)

    def main1(x):
        x1 = float(x)
        if x1 >= 0.4:
            return 1
        else:
            return 0

    def main2(x):
        x1 = float(x)
        if x1 > 0.4:
            return 1
        else:
            return 0
    df['非负'] = df['官方答复_positive_probs'].apply(main1)
    df['负面'] = df['官方答复_negative_probs'].apply(main2)
    df1 = df.groupby('主题类型').agg('sum')

    x_data = [x for x in df1.index]
    y_data = [int(y) for y in df1['非负']]
    z_data = [int(z) for z in df1['负面']]

    c = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="800px",theme=ThemeType.ESSOS))
            .add_xaxis(x_data)
            .add_yaxis("非负", y_data, stack="stack1", category_gap="50%")
            .add_yaxis("负面", z_data, stack="stack1", category_gap="50%")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=True,position="right",))
            .set_global_opts(title_opts=opts.TitleOpts(title="官方答复-各个主题情感倾向分布状况"),
                             xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),)
            .render("./data/官方答复-各个主题情感倾向分布状况.html")
    )


if __name__ == '__main__':
    plt_pie()
    plt_pie1()
    month_number()
    emotion_bar()
    emotion_bar1()
    lda_bar()
    lda_bar1()
