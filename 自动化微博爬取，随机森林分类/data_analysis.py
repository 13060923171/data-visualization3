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
from pyecharts.charts import Pie

#停用词函数
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


def map():
    new_df = data['ip属地'].value_counts()
    new_df = new_df.sort_values(ascending=False)
    provinces = ['安徽', '澳门', '北京', '重庆', '福建', '甘肃', '广东', '广西', '贵州', '海南', '河北', '黑龙江', '河南', '湖北', '湖南', '江苏', '江西',
                 '吉林', '辽宁', '内蒙古', '宁夏', '青海', '山东', '上海', '山西', '陕西', '四川', '台湾', '天津', '西藏', '香港', '新疆', '云南', '浙江']
    map_data = []
    for z,v in zip(new_df.index, new_df.values):
        if z in provinces:
            map_data.append([z,int(v)])
        else:
            pass

    c = (
        Geo()
            .add_schema(maptype="china")
            .add("ip属地", map_data)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(is_piecewise=True,max_=int(list(new_df.values)[0]) + 50,min_=0),
            title_opts=opts.TitleOpts(title="用户IP属地分布概况"),
        )
            .render("./img/用户IP属地分布概况.html")
    )


def word():
    d = {}
    for i in data['简介']:
        w = str(i).split(" ")
        for j in w:
            d[j] = d.get(j, 0) + 1
    ls = list(d.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    ls = ls[:100]

    (
        WordCloud(init_opts=opts.InitOpts(width="900px", height="800px"))
            .add(series_name="用户简介", data_pair=ls, word_size_range=[24, 55])
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title="简介热词分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
            .render("./img/简介热词分析.html")
    )


def bar1():
    df = data[data['用户分类'] == 0]
    average1 = round(df['粉丝量'].astype('int').mean(),1)
    average2 = round(df['关注量'].astype('int').mean(),1)
    average3 = round(df['博文数'].astype('int').mean(),1)
    x_data = ['粉丝量','关注量','博文数']
    y_data1 = [average1,average2,average3]

    df1 = data[data['用户分类'] == 1]
    average4 = round(df1['粉丝量'].astype('int').mean(),1)
    average5 = round(df1['关注量'].astype('int').mean(),1)
    average6 = round(df1['博文数'].astype('int').mean(),1)
    y_data2 = [average4, average5, average6]

    df1 = data[data['用户分类'] == 2]
    average4 = round(df1['粉丝量'].astype('int').mean(), 1)
    average5 = round(df1['关注量'].astype('int').mean(), 1)
    average6 = round(df1['博文数'].astype('int').mean(), 1)
    y_data3 = [average4, average5, average6]

    df1 = data[data['用户分类'] == 3]
    average4 = round(df1['粉丝量'].astype('int').mean(), 1)
    average5 = round(df1['关注量'].astype('int').mean(), 1)
    average6 = round(df1['博文数'].astype('int').mean(), 1)
    y_data4 = [average4, average5, average6]

    c = (
        Bar()
            .add_xaxis(x_data)
            .add_yaxis("机器用户", y_data1)
            .add_yaxis("低活跃用户", y_data2)
            .add_yaxis("普通用户", y_data3)
            .add_yaxis("高活跃用户", y_data4)
            .set_global_opts(
            title_opts=opts.TitleOpts(title="数值表现-均值"),
        )
            .render("./img/数值表现.html")
    )


def pie():
    def demo(x):
        x1 = int(x)
        if x1 == 0:
            return '机器用户'
        if x1 == 1:
            return '低活跃用户'
        if x1 == 2:
            return '普通用户'
        if x1 == 3:
            return '高活跃用户'
    data['用户分类'] = data['用户分类'].apply(demo)
    new_df = data['用户分类'].value_counts()
    new_df = new_df.sort_values(ascending=False)

    c = (
        Pie()
            .add(
            "用户分类",
            [(j,int(k)) for j,k in zip(new_df.index, new_df.values)],
            center=["35%", "50%"],
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="用户分布概况"),
            legend_opts=opts.LegendOpts(pos_left="15%"),
        )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}  {per|{d}%}"))
            .render("./img/用户分布概况.html")
    )


if __name__ == '__main__':
    data = pd.read_csv('./data/new_data.csv')
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
    map()
    word()
    bar1()
    pie()