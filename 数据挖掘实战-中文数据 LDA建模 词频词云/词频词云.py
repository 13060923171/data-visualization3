import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jieba
import jieba.posseg as pseg
import pyecharts.options as opts
from pyecharts.charts import WordCloud

sns.set_style(style="whitegrid")


def main1(df,name):
    # 导入停用词列表
    stop_words = []
    with open("custom_dict.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())
    # 使用自定义词典
    jieba.load_userdict("custom_dict.txt")

    def get_cut_words(content_series):
        try:
            # 对文本进行分词和词性标注
            words = pseg.cut(content_series)
            # 保存名词和形容词的列表
            nouns_and_adjs = []
            # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
            for word, flag in words:
                if word in stop_words:
                    # 如果是名词或形容词，就将其保存到列表中
                    nouns_and_adjs.append(word)
            if len(nouns_and_adjs) != 0:
                return ' '.join(nouns_and_adjs)
            else:
                return np.NAN
        except:
            return np.NAN


    df['fenci1'] = df['content'].apply(get_cut_words)
    df = df.dropna(subset=['fenci1'],axis=0)
    d = {}
    list_text = []
    for t in df['fenci1']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            # 对文本进行分词和词性标注
            # 添加到列表里面
            list_text.append(i)
            d[i] = d.get(i, 0) + 1

    ls = list(d.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    for key, values in ls:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_excel('./LDA/{}/重要词汇.xlsx'.format(name), index=False)


def word_1(df,name):
    data = []
    for i,j in zip(df['word'],df[' counts']):
        data.append((i,j))
    (
        WordCloud()
            .add(series_name="热点分析", data_pair=data, word_size_range=[10, 150])
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title="热点分析", title_textstyle_opts=opts.TextStyleOpts(font_size=24)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
            .render("./LDA/{}/wordcloud.html".format(name))
    )


if __name__ == '__main__':
    df = pd.read_excel('new_data.xlsx')
    x_data = ['新一代信息技术']
    for x in x_data:
        # df1 = df[df['所属领域'] == x]
        # main1(df1,x)
        df2 = pd.read_excel('./LDA/{}/重要词汇.xlsx'.format(x))
        word_1(df2,x)

