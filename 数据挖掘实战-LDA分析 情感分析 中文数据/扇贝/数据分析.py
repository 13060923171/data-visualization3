import pandas as pd
import numpy as np
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
from pyecharts.charts import Bar
import pyecharts.options as opts
from pyecharts.charts import Pie

matplotlib.use('Agg')  # 使用Agg后端

df1 = pd.read_excel('./扇贝-lda/扇贝-lda_data(二级维度-技术设计).xlsx')
df1['二级-主题类型'] = df1['二级-主题类型'].replace(0,'目标导向').replace(1,'易于使用').replace(2,'导航').replace(3,'交互性')
df2 = pd.read_excel('./扇贝-lda/扇贝-lda_data(二级维度-艺术审美).xlsx')
df2['二级-主题类型'] = df2['二级-主题类型'].replace(0,'学习者控制').replace(1,'实用').replace(2,'个人喜好').replace(3,'美学')

df6 = pd.concat([df1,df2],axis=0)

df3 = pd.read_excel('./扇贝-lda/特征维度(二级维度-技术设计).xlsx')
df3['所属主题'] = df3['所属主题'].replace('Topic0','目标导向').replace('Topic1','易于使用').replace('Topic2','导航').replace('Topic3','交互性')
df4 = pd.read_excel('./扇贝-lda/特征维度(二级维度-艺术审美).xlsx')
df4['所属主题'] = df4['所属主题'].replace('Topic0','学习者控制').replace('Topic1','实用').replace('Topic2','个人喜好').replace('Topic3','美学')

df5 = pd.read_excel('./扇贝-lda/特征维度(一级维度).xlsx')
df5['所属主题'] = df5['所属主题'].replace('Topic0','技术设计').replace('Topic1','艺术审美')
list_name = ['技术设计','艺术审美']

def pie1(data,name):
    x_data = [x for x in data['所属主题']]
    y_data = [y for y in data['文本数量']]

    data_pair = [list(z) for z in zip(x_data, y_data)]
    data_pair.sort(key=lambda x: x[1])

    c = (
        Pie()
            .add(
            "",
            data_pair,
            center=["35%", "50%"],
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title=f"APP-{name}"),
            legend_opts=opts.LegendOpts(pos_left="15%"),
        )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
            .render(f"APP-{name}.html")
    )


list_df = [df3,df4]

for d,n in zip(list_df,list_name):
    pie1(d,n)

data = pd.DataFrame()
data['特征二级维度'] = df3['所属主题']
data['特征二级维度关注度'] = df3['维度关注度']
data['特征二级维度-关注度内部占比'] = df3['关注度内部占比']
data['特征一级维度'] = '{}'.format(df5['所属主题'].values[0])
data['特征一级维度关注度'] = '{}'.format(df5['维度关注度'].values[0])
data['特征一级维度-关注度内部占比'] = '{}'.format(df5['内部占比'].values[0])


data1 = pd.DataFrame()
data1['特征二级维度'] = df4['所属主题']
data1['特征二级维度关注度'] = df4['维度关注度']
data1['特征二级维度-关注度内部占比'] = df4['关注度内部占比']
data1['特征一级维度'] = '{}'.format(df5['所属主题'].values[1])
data1['特征一级维度关注度'] = '{}'.format(df5['维度关注度'].values[1])
data1['特征一级维度-关注度内部占比'] = '{}'.format(df5['内部占比'].values[1])

new_data = pd.concat([data,data1],axis=0)
new_data = new_data[['特征一级维度','特征一级维度关注度','特征一级维度-关注度内部占比','特征二级维度','特征二级维度关注度','特征二级维度-关注度内部占比']]
new_data.to_excel('特征数据.xlsx',index=False)


def data_process(df,t):
    df1 = df[df['二级-主题类型'] == t]
    new_df = df1['sentiment'].value_counts()
    d = {}
    for x,y in zip(new_df.index,new_df.values):
        d[x] = y
    try:
        pos = d['正面']
    except:
        pos = 0

    try:
        neg = d['负面']
    except:
        neg = 0

    return t,pos,neg


list_name = []
list_pos = []
list_neg = []
new_df = df6['二级-主题类型'].value_counts()
type1 = [x for x in new_df.index]
for t in type1:
    t,pos,neg = data_process(df6,t)
    list_name.append(t)
    list_pos.append(int(pos))
    list_neg.append(int(neg))




# 创建堆叠条形图
bar = (
    Bar()
    .add_xaxis(list_name)
    # 添加正向数据堆叠层（颜色绿色）
    .add_yaxis("正向", list_pos, stack="stack1", color="#4DB6AC")
    # 添加负向数据堆叠层（颜色红色）
    .add_yaxis("负向", list_neg, stack="stack1", color="#c23531")
    .reversal_axis()
    # 全局配置
    .set_global_opts(
        title_opts=opts.TitleOpts(title="APP-各特征维度情感分析"),
        xaxis_opts=opts.AxisOpts(name="特征"),
        yaxis_opts=opts.AxisOpts(name="数量"),
        legend_opts=opts.LegendOpts(pos_top="5%")
    )
)

# 生成HTML文件（默认保存到当前目录）
bar.render("APP-各特征维度情感分析.html")