import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType


df1 = pd.read_csv('./data/new_纽约.csv')
df2 = pd.read_csv('./data/new_爱尔兰.csv')


def main1():
    new_df = df1['主题类型'].value_counts()
    x_data = list(new_df.index)
    y_data = list(new_df.values)

    new_df1 = df2['主题类型'].value_counts()
    x_data1 = list(new_df1.index)
    y_data1 = list(new_df1.values)

    c = (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add(
            "纽约",
            [(str(x),int(y)) for x,y in zip(x_data, y_data)],
            radius=["30%", "75%"],
            center=["25%", "50%"],
            rosetype="radius",

            # label_opts=opts.LabelOpts(is_show=False),
        )
            .add(
            "爱尔兰",
            [(str(x),int(y)) for x,y in zip(x_data1, y_data1)],
            radius=["30%", "75%"],
            center=["75%", "50%"],
            rosetype="area",
        )
            .set_global_opts(title_opts=opts.TitleOpts(title="主题对比分析"))
            .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
            ),
        )
            .render("主题对比分析.html")
    )


def main2():
    new_df = df1['comp_score'].value_counts()
    x_data = list(new_df.index)
    y_data = list(new_df.values)

    new_df1 = df2['comp_score'].value_counts()
    x_data1 = list(new_df1.index)
    y_data1 = list(new_df1.values)

    c = (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add(
            "纽约",
            [(str(x),int(y)) for x,y in zip(x_data, y_data)],
            radius=["30%", "75%"],
            center=["25%", "50%"],
            rosetype="radius",

            # label_opts=opts.LabelOpts(is_show=False),
        )
            .add(
            "爱尔兰",
            [(str(x),int(y)) for x,y in zip(x_data1, y_data1)],
            radius=["30%", "75%"],
            center=["75%", "50%"],
            rosetype="area",
        )
            .set_global_opts(title_opts=opts.TitleOpts(title="情感对比分析"))
            .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
            ),
        )
            .render("情感对比分析.html")
    )


if __name__ == '__main__':
    main1()
    main2()