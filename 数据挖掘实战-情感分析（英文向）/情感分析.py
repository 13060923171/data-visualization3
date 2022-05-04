import pandas as pd
import nltk
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
import matplotlib.pyplot as plt

df = pd.read_csv('new_twitter.csv',encoding="utf-8-sig")

df1 = df[df['translate'] == 'en']

sid = SentimentIntensityAnalyzer()
sum_counts = 0
text_list = []


def emotional_judgment(x):
    neg = x['neg']
    neu = x['neu']
    pos = x['pos']
    compound = x['compound']
    if compound == 0 and neg == 0 and pos == 0 and neu == 1:
        return 'neu'
    if compound > 0:
        if pos > neg:
            return 'pos'
        else:
            return 'neg'
    elif compound < 0:
        if pos < neg:
            return 'neg'
        else:
            return 'pos'


def time_chuli(x):
    x = str(x)
    x = x.split('-')
    x = x[1]
    return x

def total_number(x):
    df = x
    total = df['comp_score'].value_counts()
    d = {}
    for i,j in zip(total.index,total.values):
        d[i] = j
    return d


df1['scores'] = df1['new_content'].apply(lambda commentText: sid.polarity_scores(commentText))
df1['compound'] = df1['scores'].apply(lambda score_dict: score_dict['compound'])
df1['Negtive'] = df1['scores'].apply(lambda score_dict: score_dict['neg'])
df1['Postive'] = df1['scores'].apply(lambda score_dict: score_dict['pos'])
df1['Neutral'] = df1['scores'].apply(lambda score_dict: score_dict['neu'])
df1['comp_score'] = df1['scores'].apply(emotional_judgment)
df1['发表时间1'] = df1['发表时间'].apply(time_chuli)
df1['文章数量'] = 1
df1['发表时间2'] = pd.to_datetime(df1['发表时间'])
df1.index = df1['发表时间2']
new_df1 = df1['文章数量'].resample('M').sum()
plt.figure(figsize=(12,9),dpi=300)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(new_df1,color='#b82410',label='趋势')
plt.legend()
plt.title('推文数量趋势')
plt.xlabel('时间')
plt.ylabel('总数')
plt.grid()
plt.savefig('趋势图.png')
plt.show()



new_df = df1.groupby('发表时间1').apply(total_number)
x_data = list(new_df.index)
y_data = list(new_df.values)
neu_list = []
neg_list = []
pos_list = []

for y in y_data:
    try:
        neu = y['neu']
    except:
        neu = 0
    try:
        neg = y['neg']
    except:
        neg = 0
    try:
        pos = y['pos']
    except:
        pos = 0
    neu_list.append(neu)
    neg_list.append(neg)
    pos_list.append(pos)

list1 = []
list2 = []
list3 = []

for p,nu,ng in zip(pos_list,neu_list,neg_list):
    number = p / (p+nu+ng)
    number1 = '%0.2lf' % number
    number1 = float(float(number1) * 100)

    d = {
        "value": number1, "percent": number
    }
    list1.append(d)

    number2 = nu / (p + nu + ng)
    number3 = '%0.2lf' % number2
    number3 = float(float(number3) * 100)

    d1 = {
        "value": number3, "percent": number2
    }
    list2.append(d1)

    number4 = ng / (p + nu + ng)
    number5 = '%0.2lf' % number4
    number5 = float(float(number5) * 100)

    d2 = {
        "value": number5, "percent": number4
    }
    list3.append(d2)


c = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
    .add_xaxis(x_data)
    .add_yaxis("pos", list1, stack="stack1", category_gap="50%")
    .add_yaxis("neu", list2, stack="stack1", category_gap="50%")
    .add_yaxis("neg", list3, stack="stack1", category_gap="50%")
    .set_series_opts(
        label_opts=opts.LabelOpts(
            position="right",
            formatter=JsCode(
                "function(x){return Number(x.data.percent * 100).toFixed() + '%';}"
            ),
        )
    )
    .set_global_opts(
        tooltip_opts=opts.TooltipOpts(
            is_show=True, trigger="axis", axis_pointer_type="cross"
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
        ),
        yaxis_opts=opts.AxisOpts(
            name="",
            type_="value",
            min_=0,
            max_=100,
            interval=10,
            axislabel_opts=opts.LabelOpts(formatter="{value} %"),
            axistick_opts=opts.AxisTickOpts(is_show=True),
            # splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
    )
    .render("情感分布趋势.html")
)
