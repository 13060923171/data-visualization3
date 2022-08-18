import pandas as pd
from pyecharts.globals import ThemeType
from pyecharts.charts import Bar


def main1():
    df1 = pd.read_csv('./data/第一组_高频词top200.csv').iloc[:100]
    df2 = pd.read_csv('./data/第一组_neg_top_100.csv')
    df3 = pd.read_csv('./data/第一组_neu_top_100.csv')
    df4 = pd.read_csv('./data/第一组_pos_top_100.csv')

    x_data1 = list(df1['word'])
    x_data2 = list(df2['word'])
    x_data3 = list(df3['word'])
    x_data4 = list(df4['word'])

    neg_list1 = [x for x in x_data2 if x not in x_data3]
    neg_list2 = [x for x in x_data2 if x not in x_data4]

    neu_list1 = [x for x in x_data3 if x not in x_data2]
    neu_list2 = [x for x in x_data3 if x not in x_data4]

    pos_list1 = [x for x in x_data4 if x not in x_data2]
    pos_list2 = [x for x in x_data4 if x not in x_data3]

    neg = [x for x in x_data1 if x in (neg_list1+neg_list2)]
    neu = [x for x in x_data1 if x in (neu_list1+neu_list2)]
    pos = [x for x in x_data1 if x in (pos_list1+pos_list2)]

    print('ico在top100中，负面词数量为:',len(neg))
    print('ico在top100中，中立词数量为:',len(neu))
    print('ico在top100中，正面词数量为:',len(pos))


def main2():
    df1 = pd.read_csv('./data/第二组_高频词top200.csv').iloc[:100]
    df2 = pd.read_csv('./data/第二组_neg_top_100.csv')
    df3 = pd.read_csv('./data/第二组_neu_top_100.csv')
    df4 = pd.read_csv('./data/第二组_pos_top_100.csv')

    x_data1 = list(df1['word'])
    x_data2 = list(df2['word'])
    x_data3 = list(df3['word'])
    x_data4 = list(df4['word'])

    neg_list1 = [x for x in x_data2 if x not in x_data3]
    neg_list2 = [x for x in x_data2 if x not in x_data4]

    neu_list1 = [x for x in x_data3 if x not in x_data2]
    neu_list2 = [x for x in x_data3 if x not in x_data4]

    pos_list1 = [x for x in x_data4 if x not in x_data2]
    pos_list2 = [x for x in x_data4 if x not in x_data3]

    neg = [x for x in x_data1 if x in (neg_list1 + neg_list2)]
    neu = [x for x in x_data1 if x in (neu_list1 + neu_list2)]
    pos = [x for x in x_data1 if x in (pos_list1 + pos_list2)]

    print('ieo在top100中，负面词数量为:', len(neg))
    print('ieo在top100中，中立词数量为:', len(neu))
    print('ieo在top100中，正面词数量为:', len(pos))


def main3():
    df2 = pd.read_csv('./data/第一组_neg.csv')
    df3 = pd.read_csv('./data/第一组_neu.csv')
    df4 = pd.read_csv('./data/第一组_pos.csv')
    x_data2 = sum(list(df2['counts']))
    x_data3 = sum(list(df3['counts']))
    x_data4 = sum(list(df4['counts']))


    df5 = pd.read_csv('./data/第二组_neg.csv')
    df6 = pd.read_csv('./data/第二组_neu.csv')
    df7 = pd.read_csv('./data/第二组_pos.csv')

    x_data5 = sum(list(df5['counts']))
    x_data6 = sum(list(df6['counts']))
    x_data7 = sum(list(df7['counts']))

    c = (
        Bar()
            .add_xaxis(['neg','neu','pos'])
            .add_yaxis("ico", [x_data2,x_data3,x_data4],color='#85C1E9')
            .add_yaxis("ieo", [x_data5,x_data6,x_data7],color='#2980B9')
            .set_global_opts(
            title_opts={"text": "Emotional ratio"}
        )
            .render("./data/Emotional ratio.html")
    )


if __name__ == '__main__':
    # main1()
    # main2()
    main3()