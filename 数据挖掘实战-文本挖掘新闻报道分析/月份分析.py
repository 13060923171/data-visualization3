import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
from pyecharts.globals import ThemeType

df1 = pd.read_csv('./data/new_纽约.csv')
df2 = pd.read_csv('./data/new_爱尔兰.csv')


def main1(x):
    x1 = str(x).split('-')
    x1 = x1[0] + "-" + x1[1]
    return x1

df1['时间'] = df1['时间'].apply(main1)
df2['时间'] = df2['时间'].apply(main1)


def demo():
    def main2(x):
        df3 = x
        data = df3['主题类型'].value_counts()
        data.sort_index(inplace=True)
        x_data = list(data.index)
        y_data = list(data.values)
        d = []
        for x,y in zip(x_data,y_data):
            d1 = {
                "value": int(y), "percent": float(y / sum(y_data))
            }
            d.append(d1)
        return d
    new_df = df1.groupby('时间').apply(main2)
    x_data = list(new_df.index)
    y_data1 = []
    y_data2 = []
    y_data3 = []
    y_data4 = []
    y_data5 = []
    y_data6 = []
    for x in list(new_df.values):
        y_data1.append(x[0])
        y_data2.append(x[1])
        y_data3.append(x[2])
        y_data4.append(x[3])
        y_data5.append(x[4])
        y_data6.append(x[5])

    new_df1 = df2.groupby('时间').apply(main2)
    y_data11 = []
    y_data21 = []
    y_data31 = []
    y_data41 = []

    for x in list(new_df1.values):
        y_data11.append(x[0])
        y_data21.append(x[1])
        y_data31.append(x[2])
        y_data41.append(x[3])

    c = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="800px",theme=ThemeType.LIGHT))
            .add_xaxis(x_data)
            .add_yaxis("纽约主题0", y_data1, stack="stack1", category_gap="30%")
            .add_yaxis("纽约主题1", y_data2, stack="stack1", category_gap="30%")
            .add_yaxis("纽约主题2", y_data3, stack="stack1", category_gap="30%")
            .add_yaxis("纽约主题3", y_data4, stack="stack1", category_gap="30%")
            .add_yaxis("纽约主题4", y_data5, stack="stack1", category_gap="30%")
            .add_yaxis("纽约主题5", y_data6, stack="stack1", category_gap="30%")
            .add_yaxis("爱尔兰主题0", y_data11, stack="stack3", category_gap="50%")
            .add_yaxis("爱尔兰主题1", y_data21, stack="stack3", category_gap="50%")
            .add_yaxis("爱尔兰主题2", y_data31, stack="stack3", category_gap="50%")
            .add_yaxis("爱尔兰主题3", y_data41, stack="stack3", category_gap="50%")
            .set_global_opts(title_opts=opts.TitleOpts(title="主题月份对比分析"))
            .set_series_opts(
            label_opts=opts.LabelOpts(
                position="center",
                color='black',
                formatter=JsCode(
                    "function(x){return Number(x.data.percent * 100).toFixed() + '%';}"
                ),
            )
        )
            .render("主题月份对比.html")
    )


def demo2():
    def main2(x):
        df3 = x
        data = df3['comp_score'].value_counts()
        data.sort_index(inplace=True)
        if len(data.index) > 2:
            data.drop(index='neu',inplace=True)
        else:
            data = data
        x_data = list(data.index)
        y_data = list(data.values)
        d = []
        for x,y in zip(x_data,y_data):
            d1 = {
                "value": int(y), "percent": float(y / sum(y_data))
            }
            d.append(d1)
        return d
    new_df = df1.groupby('时间').apply(main2)
    x_data = list(new_df.index)
    y_data1 = []
    y_data2 = []
    for x in list(new_df.values):
        y_data1.append(x[0])
        y_data2.append(x[1])

    new_df1 = df2.groupby('时间').apply(main2)
    y_data11 = []
    y_data21 = []
    for x in list(new_df1.values):
        y_data11.append(x[0])
        y_data21.append(x[1])

    c = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="800px",theme=ThemeType.LIGHT))
            .add_xaxis(x_data)
            .add_yaxis("纽约-负面", y_data1, stack="stack1", category_gap="30%")
            .add_yaxis("纽约-正面", y_data2, stack="stack1", category_gap="30%")
            .add_yaxis("爱尔兰-负面", y_data11, stack="stack3", category_gap="50%")
            .add_yaxis("爱尔兰-正面", y_data21, stack="stack3", category_gap="50%")
            .set_global_opts(title_opts=opts.TitleOpts(title="情感月份对比分析"))
            .set_series_opts(
            label_opts=opts.LabelOpts(
                position="center",
                color='black',
                formatter=JsCode(
                    "function(x){return Number(x.data.percent * 100).toFixed() + '%';}"
                ),
            )
        )
            .render("情感月份对比.html")
    )


if __name__ == '__main__':
    demo()
    demo2()

