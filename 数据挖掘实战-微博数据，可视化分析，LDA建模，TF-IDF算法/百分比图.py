from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
import pandas as pd

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
new_df = df['发帖数量'].resample('Y').sum()

list2 = [
    {"value": (3359 + 91250+4811 + 91250), "percent": 3359 / (3359 + 91250+4811 + 91250)},
    {"value": (3359 + 91250+4811 + 91250), "percent": 4811 / (4811 + 91250+4811 + 91250)},

]

list3 = [
    {"value": (3359 + 91250+4811 + 91250), "percent": 91250 / (3359 + 91250+4811 + 91250)},
    {"value": (3359 + 91250+4811 + 91250), "percent": 91250 / (4811 + 91250+4811 + 91250)},
]

c = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
    .add_xaxis(['2021', '2022'])
    .add_yaxis("碳排放", list2, stack="stack1", category_gap="50%")
    .add_yaxis("总博文", list3, stack="stack1", category_gap="50%")
    .set_series_opts(
        label_opts=opts.LabelOpts(
            position="right",
            formatter=JsCode(
                "function(x){return Number(x.data.percent * 100).toFixed() + '%';}"
            ),
        )
    )
    .render("./data/碳排放与总博文占比趋势图.html")
)
