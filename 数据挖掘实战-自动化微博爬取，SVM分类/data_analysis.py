import pandas as pd
import numpy as np
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import scipy
from joblib import dump, load
from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.charts import WordCloud
from pyecharts.charts import Bar

def ip_chuli(x):
    x1 = str(x).split("：")
    return x1[-1]


def number_data(x):
    try:
        x1 = str(x)
        number = re.findall(r'\d+', x1)
        return number[0]
    except:
        return 0


def fan_chuli(x):
    try:
        x1 = int(x)
        return x1
    except:
        if '万' in x:
            x1 = str(x).replace('万','')
            x1 = float(x1) * 10000
            return x1


stop_words = []
with open("./data/stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


#去掉标点符号，以及机械压缩
def preprocess_word(word):
    word1 = str(word)
    word1 = re.sub(r'转发微博', '', word1)
    word1 = re.sub(r'#\w+#', '', word1)
    word1 = re.sub(r'【.*?】', '', word1)
    word1 = re.sub(r'@[\w]+', '', word1)
    word1 = re.sub(r'[a-zA-Z]', '', word1)
    word1 = re.sub(r'\.\d+', '', word1)
    return word1


def emjio_tihuan(x):
    x1 = str(x)
    x2 = re.sub('(\[.*?\])', "", x1)
    x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
    x4 = re.sub(r'\n', '', x3)
    return x4


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


# 定义机械压缩函数
def yasuo(st):
    for i in range(1, int(len(st) / 2) + 1):
        for j in range(len(st)):
            if st[j:j + i] == st[j + i:j + 2 * i]:
                k = j + i
                while st[k:k + i] == st[k + i:k + 2 * i] and k < len(st):
                    k = k + i
                st = st[:j] + st[k:]
    return st


def get_cut_words(content_series):
    # 读入停用词表
    # 分词
    word_num = jieba.lcut(content_series, cut_all=False)

    # 条件筛选
    word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

    return ' '.join(word_num_selected)


def null_paichu(x):
    x1 = str(x)
    if len(x1) != 0:
        return x1
    else:
        return np.NAN


data = pd.read_csv('./data/pred_data.csv')
data['ip属地'] = data['ip属地'].apply(ip_chuli)
data['博文数'] = data['博文数'].apply(number_data)
data['粉丝量'] = data['粉丝量'].apply(fan_chuli)
data['关注量'] = data['关注量'].apply(fan_chuli)
data['关注量'] = data['关注量'].astype('int')
data['粉丝量'] = data['粉丝量'].astype('int')
data['简介'] = data['简介'].apply(preprocess_word)
data['简介'] = data['简介'].apply(emjio_tihuan)
data['简介'] = data['简介'].apply(yasuo)
data['简介'] = data['简介'].apply(get_cut_words)
data['简介'] = data['简介'].apply(null_paichu)
data['昵称'] = data['昵称'].apply(preprocess_word)
data['昵称'] = data['昵称'].apply(emjio_tihuan)
data['昵称'] = data['昵称'].apply(yasuo)
data['昵称'] = data['昵称'].apply(get_cut_words)
data['昵称'] = data['昵称'].apply(null_paichu)
data = data.dropna(subset=['简介', '昵称'], axis=0)


def map():
    df = data[data['用户特征'] == 1]
    new_df = df['ip属地'].value_counts()
    new_df = new_df.sort_values(ascending=False)
    provinces = ['安徽', '澳门', '北京', '重庆', '福建', '甘肃', '广东', '广西', '贵州', '海南', '河北', '黑龙江', '河南', '湖北', '湖南', '江苏', '江西',
                 '吉林', '辽宁', '内蒙古', '宁夏', '青海', '山东', '上海', '山西', '陕西', '四川', '台湾', '天津', '西藏', '香港', '新疆', '云南', '浙江']
    map_data = []
    for z,v in zip(new_df.index, new_df.values):
        if z in provinces:
            map_data.append([z,int(v)])
        else:
            pass

    df1 = data[data['用户特征'] == 0]
    new_df1 = df1['ip属地'].value_counts()
    new_df1 = new_df1.sort_values(ascending=False)
    map_data1 = []
    for z, v in zip(new_df1.index, new_df1.values):
        if z in provinces:
            map_data1.append([z, int(v)])
        else:
            pass

    c = (
        Geo()
            .add_schema(maptype="china")
            .add("真实用户", map_data)
            .add("机器人", map_data1)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(is_piecewise=True,max_=int(list(new_df.values)[0]) + 50,min_=0),
            title_opts=opts.TitleOpts(title="IP属地"),
        )
            .render("./data/geo_visualmap_piecewise.html")
    )


def word1():
    df = data[data['用户特征'] == 1]
    d = {}
    for i in df['昵称']:
        w = str(i).split(" ")
        for j in w:
            d[j] = d.get(j,0)+1
    ls = list(d.items())
    ls.sort(key=lambda x:x[1],reverse=True)
    ls = ls[:100]

    df1 = data[data['用户特征'] == 0]
    d1 = {}
    for i in df1['昵称']:
        w = str(i).split(" ")
        for j in w:
            d1[j] = d1.get(j, 0) + 1
    ls1 = list(d1.items())
    ls1.sort(key=lambda x: x[1], reverse=True)
    ls1 = ls1[:50]

    (
        WordCloud()
            .add(series_name="真实用户", data_pair=ls, word_size_range=[6, 66])
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title="昵称分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
            .render("./data/basic_wordcloud1.html")
    )

    (
        WordCloud()
            .add(series_name="机器人", data_pair=ls1, word_size_range=[6, 66])
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title="昵称分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
            .render("./data/basic_wordcloud2.html")
    )


def word2():
    df = data[data['用户特征'] == 1]
    d = {}
    for i in df['简介']:
        w = str(i).split(" ")
        for j in w:
            d[j] = d.get(j, 0) + 1
    ls = list(d.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    ls = ls[:100]

    df1 = data[data['用户特征'] == 0]
    d1 = {}
    for i in df1['简介']:
        w = str(i).split(" ")
        for j in w:
            d1[j] = d1.get(j, 0) + 1
    ls1 = list(d1.items())
    ls1.sort(key=lambda x: x[1], reverse=True)
    ls1 = ls1[:50]

    (
        WordCloud()
            .add(series_name="真实用户", data_pair=ls, word_size_range=[6, 66])
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title="简介分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
            .render("./data/basic_wordcloud3.html")
    )

    (
        WordCloud()
            .add(series_name="机器人", data_pair=ls1, word_size_range=[6, 66])
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title="简介分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
            .render("./data/basic_wordcloud4.html")
    )


def bar1():
    df = data[data['用户特征'] == 1]
    average1 = round(df['粉丝量'].astype('int').mean(),2)
    average2 = round(df['关注量'].astype('int').mean(),2)
    average3 = round(df['博文数'].astype('int').mean(),2)
    x_data = ['粉丝量','关注量','博文数']
    y_data1 = [average1,average2,average3]

    df1 = data[data['用户特征'] == 0]
    average4 = round(df1['粉丝量'].astype('int').mean(),2)
    average5 = round(df1['关注量'].astype('int').mean(),2)
    average6 = round(df1['博文数'].astype('int').mean(),2)
    y_data2 = [average4, average5, average6]

    c = (
        Bar()
            .add_xaxis(x_data)
            .add_yaxis("真实用户", y_data1)
            .add_yaxis("机器人", y_data2)
            .set_global_opts(
            title_opts=opts.TitleOpts(title="数值表现"),
        )
            .render("./data/bar_rotate_xaxis_label.html")
    )


if __name__ == '__main__':
    map()
    word1()
    word2()
    bar1()
