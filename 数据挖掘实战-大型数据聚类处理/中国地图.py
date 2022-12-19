import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.globals import ThemeType

df = pd.read_excel('demo-最终版.xlsx')


def main1(x):
    x1 = str(x).split('-')
    x1 = x1[0].replace('省','').replace('市','')
    x1 = x1.strip(' ')
    return x1


df['所在地区'] = df['所在地区'].apply(main1)
new_df = df['所在地区'].value_counts()

list1 = []
for i in zip(new_df.index,new_df.values):
    list1.append([i[0],int(i[1])])


c = (
    Map(init_opts=opts.InitOpts(theme=ThemeType.ESSOS))
    .add("", list1, "china")
    .set_global_opts(
        title_opts=opts.TitleOpts(title="区域分布情况"),
        visualmap_opts=opts.VisualMapOpts(max_=20000, is_piecewise=True),
    )
    .render("map.html")
)
