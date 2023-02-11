import pandas as pd
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Bar,Line,Pie
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
    df = pd.read_csv('./data/nlp_all_data.csv',parse_dates=['提问时间'], index_col="提问时间")
    df5 = pd.read_csv('./data/nlp_all_data.csv', parse_dates=['答复时间'], index_col="答复时间")
    df = df.dropna(subset=['提问内容'], axis=0)
    df5 = df5.dropna(subset=['官方答复'], axis=0)
    df['发布数据'] = 1
    df5['发布数据'] = 1

    df1 = df['发布数据'].resample('M').sum()

    x_data = [str(x).split(" ")[0] for x in df1.index]
    y_data = [int(y) for y in df1.values]

    df3 = df5['发布数据'].resample('M').sum()
    x_data1 = [str(x).split(" ")[0] for x in df3.index][:-1]
    y_data1 = [int(y) for y in df3.values][:-1]

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


def length_emotion():
    df = pd.read_csv('./data/nlp_all_data.csv')
    df1 = df.groupby('问题领域').agg('mean')
    def sswr(x):
        x1 = round(x)
        return x1

    def sswr1(x):
        x1 = round(x,2)
        return x1
    df1['官方答复_positive_probs'] = df1['官方答复_positive_probs'].apply(sswr1)
    df1['提问内容_positive_probs'] = df1['提问内容_positive_probs'].apply(sswr1)
    df1['提问内容_长度'] = df1['提问内容_长度'].apply(sswr)
    df1['官方答复_长度'] = df1['官方答复_长度'].apply(sswr)
    x_data = list(df1.index)
    y_data1 = [y for y in df1['提问内容_长度']]
    y_data2 = [y for y in df1['官方答复_长度']]
    y_data3 = [y for y in df1['提问内容_positive_probs']]
    y_data4 = [y for y in df1['官方答复_positive_probs']]

    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.ESSOS))
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis(
            series_name="提问内容_长度",
            y_axis=y_data1,
            label_opts=opts.LabelOpts(is_show=True),
        )
            .add_yaxis(
            series_name="官方答复_长度",
            y_axis=y_data2,
            label_opts=opts.LabelOpts(is_show=True),
        )
            .extend_axis(
            yaxis=opts.AxisOpts(
                name="分值",
                type_="value",
                min_=0,
                max_=0.5,
                interval=0.1,
                axislabel_opts=opts.LabelOpts(formatter="{value} "),
            )
        )
            .set_global_opts(
            tooltip_opts=opts.TooltipOpts(
                is_show=True, trigger="axis", axis_pointer_type="cross"
            ),
            title_opts=opts.TitleOpts(title="公众政治诉求表达的文本长度和情感倾向", pos_left="center", pos_top="15"),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
            ),
            yaxis_opts=opts.AxisOpts(
                name="长度",
                type_="value",
                min_=0,
                max_=500,
                interval=100,
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
    )

    line = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.ESSOS))
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis(
            series_name="提问内容-正面走势",
            yaxis_index=1,
            y_axis=y_data3,
            label_opts=opts.LabelOpts(is_show=True),
        )
            .add_yaxis(
            series_name="官方答复-正面走势",
            yaxis_index=1,
            y_axis=y_data4,
            label_opts=opts.LabelOpts(is_show=True),
        )
    )

    bar.overlap(line).render("./data/公众政治诉求表达的文本长度和情感倾向.html")


def days_pie():
    df = pd.read_csv('./data/nlp_all_data.csv')
    def main(x):
        x1 = int(x)
        if x1 <= 7:
            return '7天内'
        elif 7 < x1 <= 14:
            return '14天内'
        elif 14 < x1 <= 21:
            return '21天内'
        elif 21 < x1 <= 28:
            return '28天内'
        else:
            return '超过1个月'
    df['间隔类型'] = df['相差间隔'].apply(main)
    new_df = df['间隔类型'].value_counts()
    x_data = list(new_df.index)
    y_data = list(new_df.values)

    c = (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.ESSOS))
            .add("", [(x,int(y)) for x,y in zip(x_data,y_data)])
            .set_global_opts(title_opts=opts.TitleOpts(title="政府回应网络民意时效情况"))
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
            .render("./data/政府回应网络民意时效情况.html")
    )


def tiwen_bar():
    df = pd.read_csv('./data/nlp_all_data.csv')
    df['数量'] = 1
    df1 = df.groupby('提问类型').agg('mean')
    df2 = df.groupby('提问类型').agg('sum')
    def sswr(x):
        x1 = round(x)
        return x1

    df1['相差间隔'] = df1['相差间隔'].apply(sswr)
    x_data = list(df1.index)
    y_data1 = [y for y in df1['相差间隔']]
    y_data2 = [y for y in df2['数量']]

    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.ESSOS))
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis(
            series_name="数量",
            y_axis=y_data2,
            label_opts=opts.LabelOpts(is_show=True),
        )
            .extend_axis(
            yaxis=opts.AxisOpts(
                name="时长",
                type_="value",
                min_=0,
                max_=25,
                interval=5,
                axislabel_opts=opts.LabelOpts(formatter="{value} 天"),
            )
        )
            .set_global_opts(
            tooltip_opts=opts.TooltipOpts(
                is_show=True, trigger="axis", axis_pointer_type="cross"
            ),
            title_opts=opts.TitleOpts(title="不同类别政府回应平均用时",pos_left="center",pos_top="15"),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
            ),
            yaxis_opts=opts.AxisOpts(
                name="数量",
                type_="value",
                min_=0,
                max_=5000,
                interval=500,
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
    )

    line = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.ESSOS))
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis(
            series_name="时长",
            yaxis_index=1,
            y_axis=y_data1,
            label_opts=opts.LabelOpts(is_show=True),
        )
    )

    bar.overlap(line).render("./data/不同类别政府回应平均用时.html")


def lyu_bar():
    df = pd.read_csv('./data/nlp_all_data.csv')
    df['数量'] = 1
    df1 = df.groupby('问题领域').agg('mean')
    df2 = df.groupby('问题领域').agg('sum')
    def sswr(x):
        x1 = round(x)
        return x1

    df1['相差间隔'] = df1['相差间隔'].apply(sswr)
    x_data = list(df1.index)
    y_data1 = [y for y in df1['相差间隔']]
    y_data2 = [y for y in df2['数量']]

    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.ESSOS))
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis(
            series_name="数量",
            y_axis=y_data2,
            label_opts=opts.LabelOpts(is_show=True),
        )
            .extend_axis(
            yaxis=opts.AxisOpts(
                name="时长",
                type_="value",
                min_=0,
                max_=30,
                interval=5,
                axislabel_opts=opts.LabelOpts(formatter="{value} 天"),
            )
        )
            .set_global_opts(
            tooltip_opts=opts.TooltipOpts(
                is_show=True, trigger="axis", axis_pointer_type="cross"
            ),
            title_opts=opts.TitleOpts(title="不同类别政府回应平均用时",pos_left="center",pos_top="15"),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
            ),
            yaxis_opts=opts.AxisOpts(
                name="数量",
                type_="value",
                min_=0,
                max_=3000,
                interval=500,
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
    )

    line = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.ESSOS))
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis(
            series_name="时长",
            yaxis_index=1,
            y_axis=y_data1,
            label_opts=opts.LabelOpts(is_show=True),
        )
    )

    bar.overlap(line).render("./data/不同领域政府回应平均用时.html")


if __name__ == '__main__':
    plt_pie()
    plt_pie1()
    month_number()
    emotion_bar()
    emotion_bar1()
    lda_bar()
    lda_bar1()
    length_emotion()
    days_pie()
    tiwen_bar()
    lyu_bar()
