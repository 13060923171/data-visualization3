import codecs
import networkx as nx
import matplotlib.pyplot as plt
import csv
import pandas as pd
import nltk
from collections import Counter
from scipy.sparse import coo_matrix
from tqdm import tqdm
import numpy as np
import re
import jieba
from tqdm import tqdm
import concurrent.futures
import os

from collections import defaultdict
from itertools import combinations
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer

from pyecharts.charts import Bar, Line
from pyecharts import options as opts
from pyecharts.globals import ThemeType

import matplotlib.pyplot as plt
import matplotlib

def main():
    df = pd.read_excel('new_data.xlsx')

    # 导入停用词列表
    keyword_words = []
    with open("keyword.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            keyword_words.append(line.strip())

    def fenci():
        f = open('fenci.txt', 'w', encoding='utf-8-sig')
        for line in df['content']:
            line = str(line)
            line = line.strip('\n')
            # 计算关键词
            all_words = line.split()
            c = Counter()
            for x in all_words:
                if x in keyword_words:
                    c[x] += 1
            output = ""
            for (k, v) in c.most_common():
                output += k + " "

            f.write(output + "\n")
        else:
            f.close()

    fenci()

    word = [] #记录关键词
    f = open("fenci.txt", encoding='utf-8')
    line = f.readline()
    while line:
        #print line
        line = line.replace("\n", "") #过滤换行
        line = line.strip('\n')
        for n in line.split(' '):
            #print n
            if n not in word:
                word.append(n)
        line = f.readline()
    f.close()

    word_vector = coo_matrix((len(word),len(word)), dtype=np.int8).toarray()
    print(word_vector.shape)

    f = open("fenci.txt", encoding='utf-8')
    line = f.readline()
    while line:
        line = line.replace("\n", "")  # 过滤换行
        line = line.strip('\n')  # 过滤换行
        nums = line.split(' ')
        # 循环遍历关键词所在位置 设置word_vector计数
        i = 0
        j = 0
        while i < len(nums):  # ABCD共现 AB AC AD BC BD CD加1
            j = i + 1
            w1 = nums[i]  # 第一个单词
            while j < len(nums):
                w2 = nums[j]  # 第二个单词
                # 从word数组中找到单词对应的下标
                k = 0
                n1 = 0
                while k < len(word):
                    if w1 == word[k]:
                        n1 = k
                        break
                    k = k + 1
                # 寻找第二个关键字位置
                k = 0
                n2 = 0
                while k < len(word):
                    if w2 == word[k]:
                        n2 = k
                        break
                    k = k + 1

                # 重点: 词频矩阵赋值 只计算上三角
                if n1 <= n2:
                    word_vector[n1][n2] = word_vector[n1][n2] + 1
                else:
                    word_vector[n2][n1] = word_vector[n2][n1] + 1
                j = j + 1
            i = i + 1
        # 读取新内容
        line = f.readline()
    f.close()

    words = codecs.open("word_node.txt", "w", "utf-8")
    i = 0
    while i < len(word):
        student1 = word[i]
        j = i + 1
        while j < len(word):
            student2 = word[j]
            if word_vector[i][j]>0:
                words.write(student1 + " " + student2 + " "
                    + str(word_vector[i][j]) + "\r\n")
            j = j + 1
        i = i + 1
    words.close()

    """ 第四步:图形生成 """
    with open('word_node.txt','r',encoding='utf-8')as f:
        content = f.readlines()
    list_word1 = []
    list_word2 = []
    list_weight = []
    for i in content:
        c = i.strip('\n').split(" ")
        list_word1.append(c[0])
        list_word2.append(c[1])
        list_weight.append(c[2])

    df = pd.DataFrame()
    df['word1'] = list_word1
    df['word2'] = list_word2
    df['weight'] = list_weight
    df['weight'] = df['weight'].astype(int)
    df = df.sort_values(by=['weight'],ascending=False)
    df = df.dropna(how='any',axis=1)
    new_df = df

    A = []
    B = []
    for w1,w2 in tqdm(zip(new_df['word1'],new_df['word2'])):
        if w1 != "" and w2 != "":
            A.append(w1)
            B.append(w2)
    elem_dic = tuple(zip(A,B))
    print(len(elem_dic))
    #创建一个空的无向图。即创建了一个称为G的图对象，用于保存文本数据的节点和边信息。
    G = nx.Graph()
    #向图G中添加节点和边。这里的list(elem_dic)表示将elem_dic字典中的元素列表作为图的边。其中elem_dic字典中存储着文本数据的节点和边信息。
    G.add_edges_from(list(elem_dic))
    #设置图像中使用中文字体，以避免出现显示中文乱码的情况。这里将字体设置为SimHei，使用sans-serif字体族。
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'sans-serif'
    #设置图像的大小，其中figsize参数设置图像的宽度和高度。
    plt.figure(figsize=(16, 9),dpi=500)
    #确定节点布局。这里使用了一种称为spring layout的布局算法，相当于在二维空间中对节点进行排列。iterations参数指定了进行节点排列的迭代次数。
    pos=nx.spring_layout(G,iterations=10)
    #绘制节点。其中alpha参数设置节点的透明度，node_size参数设置节点的大小。
    nx.draw_networkx_nodes(G, pos, alpha=0.7,node_size=800)
    #绘制边。其中width参数设置边的宽度，alpha参数设置边的透明度，edge_color参数设置边的颜色。
    nx.draw_networkx_edges(G,pos,width=0.5,alpha=0.8,edge_color='g')
    #添加标签。其中font_family参数指定图像中使用sans-serif字体族，alpha参数设置节点标签的透明度，font_size参数设置归纳节点标签的字体大小。
    nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1,font_size='10')
    plt.title("co-occurrence semantics")
    plt.savefig('共现语义.png')
    # plt.show()


def main2():
    # 导入停用词列表
    # 使用默认字典来存储词对共现频率
    co_occur = defaultdict(int)

    keyword_words = []
    with open("keyword.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            keyword_words.append(line.strip())
    corpus = []
    with open('fenci.txt','r',encoding='utf-8-sig') as f:
        content = f.readlines()
    for c in content:
        c = str(c).strip('\n')
        words = c.split()
        corpus.append(words)
        # 计算所有词对的共现频率
        for word1, word2 in combinations(words, 2):
            co_occur[(word1, word2)] += 1
            co_occur[(word2, word1)] += 1  # 如果是无向网络，还需要统计反向共现

    # 将语料库从列表列表转换为字符串列表以供TF-IDF Vectorizer使用
    corpus = [' '.join(doc) for doc in corpus]

    # 通过scikit-learn的TfidfVectorizer来计算TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # 获取每个词的IDF值
    idf = vectorizer.idf_
    word_idf_dict = dict(zip(vectorizer.get_feature_names_out(), idf))

    # 对于每个词，我们可以通过计算其在词对中出现的所有共现频率之和来计算其共现强度
    # 现在使用TF-IDF替换频率
    # strengths = defaultdict(int)
    # for (word1, word2), freq in co_occur.items():
    #     # 将频率换成TF-IDF
    #     strengths[word1] += freq / word_idf_dict.get(word1, 1)  # 如果这个词没有出现在词典中，我们就当它的IDF为1

    strengths = defaultdict(int)
    word_count = defaultdict(int)  # 用于统计每个词出现在词对中的次数
    for (word1, word2), freq in co_occur.items():
        strengths[word1] += freq
        strengths[word2] += freq
        word_count[word1] += 1
        word_count[word2] += 1

    for word in strengths:
        avg_freq = strengths[word] / word_count[word]  # 计算平均共现频率
        strengths[word] = avg_freq / word_idf_dict.get(word, 1)  # 乘以 TF-IDF
    list_values = []

    for key in keyword_words:
        values = strengths[key]
        list_values.append([key,int(values)])

    list_values.sort(key=lambda x:x[1],reverse=False)
    x_data = []
    y_data = []
    for key,value in list_values:
        x_data.append(key)
        y_data.append(value)

    c = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="900px",theme=ThemeType.MACARONS))
            .add_xaxis(x_data)
            .add_yaxis("", y_data)
            .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
            title_opts=opts.TitleOpts(title="Keyword co-occurrence intensity"),
        )
            .render("Keyword co-occurrence intensity.html")
    )


def main3():
    df1 = pd.read_excel('Untitled spreadsheet.xlsx')
    df2 = pd.read_excel('20231019015915065.xlsx')
    df3 = pd.DataFrame()
    df3['公开(公告)号'] = df2['公开(公告)号']
    df3['公开(公告)年'] = df2['公开(公告)年']
    df = pd.merge(df1, df3, on='公开(公告)号')
    df = df.drop_duplicates(subset=['摘要(译)(简体中文)'])
    df = df.drop_duplicates(subset=['标题(译)(简体中文)'])
    new_df = df['公开(公告)年'].value_counts()
    new_df = new_df.sort_index()
    x_data = [str(x) for x in new_df.index]
    y_data = [int(y) for y in new_df.values]

    c = (
        Bar(init_opts=opts.InitOpts(width="1600px", height="900px", theme=ThemeType.MACARONS))
            .add_xaxis(x_data)
            .add_yaxis("", y_data)
            .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
            title_opts=opts.TitleOpts(title="Number of patent applications"),
        )
            .render("1986-2023 year Number of patent applications.html")
    )


def main4():
    df = pd.read_excel('new_data.xlsx')
    keyword_words = []
    with open("keyword.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            keyword_words.append(line.strip())

    def time_date(time_number):
        data = df[df['time_date'] == time_number]
        co_occur = defaultdict(int)
        corpus = []
        def demo1(x):
            str1 = []
            x1 = str(x).split(" ")
            for i in x1:
                if i in keyword_words:
                    str1.append(i)
            if len(str1) != 0:
                return ' '.join(str1)
            else:
                return np.NAN

        data['content'] = data['content'].apply(demo1)
        data = data.dropna(subset=['content'],axis=0)
        for c in data['content']:
            c = str(c).strip('\n')
            words = c.split()
            corpus.append(words)
            # 计算所有词对的共现频率
            for word1, word2 in combinations(words, 2):
                co_occur[(word1, word2)] += 1
                co_occur[(word2, word1)] += 1  # 如果是无向网络，还需要统计反向共现


        # 将语料库从列表列表转换为字符串列表以供TF-IDF Vectorizer使用
        corpus1 = []
        for doc in corpus:
            if len(doc) != 0:
                corpus1.append(' '.join(doc))
            else:
                pass


        # 通过scikit-learn的TfidfVectorizer来计算TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus1)

        # 获取每个词的IDF值
        idf = vectorizer.idf_
        word_idf_dict = dict(zip(vectorizer.get_feature_names_out(), idf))

        # 对于每个词，我们可以通过计算其在词对中出现的所有共现频率之和来计算其共现强度
        # 现在使用TF-IDF替换频率
        # strengths = defaultdict(int)
        # for (word1, word2), freq in co_occur.items():
        #     # 将频率换成TF-IDF
        #     strengths[word1] += freq / word_idf_dict.get(word1, 1)  # 如果这个词没有出现在词典中，我们就当它的IDF为1

        strengths = defaultdict(int)
        word_count = defaultdict(int)  # 用于统计每个词出现在词对中的次数
        for (word1, word2), freq in co_occur.items():
            strengths[word1] += freq
            strengths[word2] += freq
            word_count[word1] += 1
            word_count[word2] += 1

        for word in strengths:
            avg_freq = strengths[word] / word_count[word]  # 计算平均共现频率
            strengths[word] = avg_freq / word_idf_dict.get(word, 1)  # 乘以 TF-IDF

        d = {}
        for key in keyword_words[:10]:
            values = strengths[key]
            d[key] = int(values)

        return d
    date = [2018,2019,2020,2021,2022,2023]
    list_d = []
    for d in date:
        dic = time_date(d)
        list_d.append(dic)
    key1 = keyword_words[:10]
    values1 = []
    for key in keyword_words[:10]:
        values2 = []
        for l in list_d:
            values = l[key]
            values2.append(int(values))
        values1.append(values2)
        
    (
        Line(init_opts=opts.InitOpts(width="1600px", height="800px", theme=ThemeType.MACARONS))
            .add_xaxis([str(x) for x in date])
            .add_yaxis(
            series_name="{}".format(key1[0]),
            symbol="emptyCircle",
            is_symbol_show=False,
            y_axis=values1[0],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
            .add_yaxis(
            series_name="{}".format(key1[1]),
            symbol="emptyCircle",
            is_symbol_show=False,
            y_axis=values1[1],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
            .add_yaxis(
            series_name="{}".format(key1[2]),
            symbol="emptyCircle",
            is_symbol_show=False,
            y_axis=values1[2],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
            .add_yaxis(
            series_name="{}".format(key1[3]),
            symbol="emptyCircle",
            is_symbol_show=False,
            y_axis=values1[3],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
            .add_yaxis(
            series_name="{}".format(key1[4]),
            symbol="emptyCircle",
            is_symbol_show=False,
            y_axis=values1[4],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
            .add_yaxis(
            series_name="{}".format(key1[5]),
            symbol="emptyCircle",
            is_symbol_show=False,
            y_axis=values1[5],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
            .add_yaxis(
            series_name="{}".format(key1[6]),
            symbol="emptyCircle",
            is_symbol_show=False,
            y_axis=values1[6],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
            .add_yaxis(
            series_name="{}".format(key1[7]),
            symbol="emptyCircle",
            is_symbol_show=False,
            y_axis=values1[7],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
            .add_yaxis(
            series_name="{}".format(key1[8]),
            symbol="emptyCircle",
            is_symbol_show=False,
            y_axis=values1[8],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
            .add_yaxis(
            series_name="{}".format(key1[9]),
            symbol="emptyCircle",
            is_symbol_show=False,
            y_axis=values1[9],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="top0-10 keyword evolution trends"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False, axisline_opts=opts.AxisLineOpts(
                is_on_zero=False,
            )),
        )
            .render("2018-2023 top0-10 keyword evolution trends.html")
    )


def main5():
    def plt_pie(time_number):
        df = pd.read_excel('new_data.xlsx')
        data = df[df['time_date'] == time_number]
        plt.style.use('ggplot')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure(dpi=500)
        new_df = data['主题类型'].value_counts()

        x_data = [x for x in new_df.index]
        y_data = [x for x in new_df.values]
        plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
        plt.title('{} year theme strength'.format(time_number))
        plt.tight_layout()
        plt.savefig('{} year theme strength.png'.format(time_number))

    date = [2018, 2019, 2020, 2021, 2022, 2023]
    for d in date:
        plt_pie(d)


if __name__ == '__main__':
    # main()
    # main2()
    # main3()
    # main4()
    main5()