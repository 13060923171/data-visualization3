import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pyecharts import options as opts
from pyecharts.charts import Geo

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim.models import LdaModel
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import pyLDAvis.gensim
import pyLDAvis

from collections import Counter
import itertools

from tqdm import tqdm

def user_analyze():
    df2 = pd.read_csv('一级评论数据.csv')
    df3 = pd.read_csv('二级评论数据.csv')

    new_df2 = df2['评论性别'].value_counts()
    new_df3 = df3['二级评论性别'].value_counts()

    new_df4 = new_df2 + new_df3
    new_df4.to_csv('用户性别分析.csv',encoding='utf-8-sig')
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9,6),dpi=500)

    x_data = [str(x) for x in new_df4.index]
    y_data = [int(x) for x in new_df4.values]

    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('性别分类')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('性别分类.png')


def map_analyze():
    df1 = pd.read_csv('博文数据.csv')
    df2 = pd.read_csv('一级评论数据.csv')
    df3 = pd.read_csv('二级评论数据.csv')

    new_df1 = df1['发博用户ip'].value_counts()
    new_df2 = df2['评论ip'].value_counts()
    new_df3 = df3['二级评论ip'].value_counts()
    list_d = []
    d1 = {}
    for x,y in zip(new_df1.index,new_df1.values):
        d1[x] = y
    list_d.append(d1)
    d2 = {}
    for x, y in zip(new_df2.index, new_df2.values):
        d2[x] = y
    list_d.append(d2)
    d3 = {}
    for x, y in zip(new_df3.index, new_df3.values):
        d3[x] = y
    list_d.append(d3)

    result = {}
    for d in list_d:
        for key, value in d.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value

    ls = list(result.items())
    ls.sort(key=lambda x:x[1],reverse=True)

    data = []
    provinces = ['安徽', '澳门', '北京', '重庆', '福建', '甘肃', '广东', '广西', '贵州', '海南', '河北', '黑龙江', '河南', '湖北', '湖南', '江苏', '江西',
                 '吉林', '辽宁', '内蒙古', '宁夏', '青海', '山东', '上海', '山西', '陕西', '四川', '台湾', '天津', '西藏', '香港', '新疆', '云南', '浙江']
    x_data = []
    y_data = []
    for key,values in ls[1:]:
        if key in provinces:
            data.append((key,int(values)))
            x_data.append(key)
            y_data.append(values)

    df = pd.DataFrame()
    df['area'] = x_data
    df['values'] = y_data

    df.to_csv('地区分析.csv',encoding='utf-8-sig')

    c = (
        Geo()
            .add_schema(maptype="china")
            .add(" ", data)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(
                is_piecewise=True,
                pieces=[
                    {"min": 0, "max": 300, "label": "0-300", "color": "#000000"},
                    {"min": 300, "max": 600, "label": "300-600", "color": "#808080"},
                    {"min": 600, "max": 1200, "label": "600-1200", "color": "#FFFFFF"}
                ]
            ),
            title_opts=opts.TitleOpts(title="地域分析"),
        )
            .render("地域分析.html")
    )

def emotion_analyze1():
    df2 = pd.read_csv('一级评论数据.csv')
    df3 = pd.read_csv('二级评论数据.csv')

    new_df2 = df2['一级情感类型'].value_counts()
    new_df3 = df3['二级情感类型'].value_counts()

    new_df4 = new_df2 + new_df3

    new_df4.to_csv('整体情感分析.csv',encoding='utf-8-sig')

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6), dpi=500)

    x_data = [str(x) for x in new_df4.index]
    y_data = [int(x) for x in new_df4.values]

    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('情感分析')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('情感分析.png')

def emotion_analyze2():
    df2 = pd.read_csv('一级评论数据.csv')
    df22 = df2[df2['评论性别'] == '女']
    df3 = pd.read_csv('二级评论数据.csv')
    df33 = df3[df3['二级评论性别'] == '女']

    new_df2 = df22['一级情感类型'].value_counts()
    new_df3 = df33['二级情感类型'].value_counts()

    new_df4 = new_df2 + new_df3
    new_df4.to_csv('女性情感分析.csv', encoding='utf-8-sig')
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6), dpi=500)

    x_data = [str(x) for x in new_df4.index]
    y_data = [int(x) for x in new_df4.values]

    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('女性_情感分析')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('女性_情感分析.png')

def emotion_analyze3():
    df2 = pd.read_csv('一级评论数据.csv')
    df22 = df2[df2['评论性别'] == '男']
    df3 = pd.read_csv('二级评论数据.csv')
    df33 = df3[df3['二级评论性别'] == '男']

    new_df2 = df22['一级情感类型'].value_counts()
    new_df3 = df33['二级情感类型'].value_counts()

    new_df4 = new_df2 + new_df3
    new_df4.to_csv('男性情感分析.csv', encoding='utf-8-sig')
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6), dpi=500)

    x_data = [str(x) for x in new_df4.index]
    y_data = [int(x) for x in new_df4.values]

    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('男性_情感分析')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('男性_情感分析.png')



def emotion_analyze4(name):
    df2 = pd.read_csv('一级评论数据.csv')
    df3 = pd.read_csv('二级评论数据.csv')

    central_area = ['河南', '湖北', '湖南', '江西', '安徽', '山西']
    west_area = ['内蒙古', '新疆', '宁夏', '甘肃', '青海', '陕西', '四川', '重庆', '贵州', '云南', '西藏']
    east_area = ['北京', '天津', '河北', '山东', '江苏', '上海', '浙江', '福建', '广东', '海南', '广西', '香港', '澳门', '台湾', '黑龙江', '辽宁', '吉林']

    def area(x):
        x1 = str(x)
        if x1 in central_area:
            return "中部地区"
        elif x1 in west_area:
            return "西部地区"
        elif x1 in east_area:
            return "东部地区"
        else:
            return "其他地区"

    df2['评论ip'] = df2['评论ip'].apply(area)
    df3['二级评论ip'] = df3['二级评论ip'].apply(area)

    df22 = df2[df2['评论ip'] == name]
    df33 = df3[df3['二级评论ip'] == name]

    new_df2 = df22['一级情感类型'].value_counts()
    new_df3 = df33['二级情感类型'].value_counts()


    new_df4 = new_df2 + new_df3
    new_df4.to_csv('{}情感分析.csv'.format(name), encoding='utf-8-sig')
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(9, 6), dpi=500)

    x_data = [str(x) for x in new_df4.index]
    y_data = [int(x) for x in new_df4.values]

    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('{}_情感分析'.format(name))
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('{}_情感分析.png'.format(name))

def influence_analyze1():
    df1 = pd.read_csv('博文数据.csv')
    time_df1 = df1.groupby('发博时间').agg('sum')

    df = pd.DataFrame()
    df['博文点赞'] = time_df1['博文点赞']
    df['博文评论'] = time_df1['博文评论']
    df['博文转发'] = time_df1['博文转发']

    df.to_csv('博文时间数据.csv',encoding='utf-8-sig')

    new_df = time_df1['博文点赞']
    x_data = [str(x) for x in new_df.index]
    y_data = [int(x) for x in new_df.values]

    new_df1 = time_df1['博文评论']
    x_data1 = [str(x) for x in new_df1.index]
    y_data1 = [int(x) for x in new_df1.values]

    new_df2 = time_df1['博文转发']
    x_data2 = [str(x) for x in new_df2.index]
    y_data2 = [int(x) for x in new_df2.values]

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(15, 12), dpi=500)

    plt.plot(x_data, y_data, color='#b82410', label='点赞')
    plt.plot(x_data1, y_data1, '^--', color='#2614e8', label='评论')
    plt.plot(x_data2, y_data2, 'go--', label='转发')
    plt.legend()
    plt.title('博文影响力趋势')
    plt.xlabel('Day')
    plt.ylabel('values')
    plt.grid()
    plt.xticks(rotation=45)
    plt.savefig('博文影响力趋势.png')
    plt.show()


def area_influence_analyze1(name):
    df1 = pd.read_csv('博文数据.csv')
    central_area = ['河南', '湖北', '湖南', '江西', '安徽', '山西']
    west_area = ['内蒙古', '新疆', '宁夏', '甘肃', '青海', '陕西', '四川', '重庆', '贵州', '云南', '西藏']
    east_area = ['北京', '天津', '河北', '山东', '江苏', '上海', '浙江', '福建', '广东', '海南', '广西', '香港', '澳门', '台湾', '黑龙江', '辽宁', '吉林']

    def area(x):
        x1 = str(x)
        if x1 in central_area:
            return "中部地区"
        elif x1 in west_area:
            return "西部地区"
        elif x1 in east_area:
            return "东部地区"
        else:
            return "其他地区"

    df1['发博用户ip'] = df1['发博用户ip'].apply(area)

    df1 = df1[df1['发博用户ip'] == name]

    time_df1 = df1.groupby('发博时间').agg('sum')

    df = pd.DataFrame()
    df['博文点赞'] = time_df1['博文点赞']
    df['博文评论'] = time_df1['博文评论']
    df['博文转发'] = time_df1['博文转发']

    df.to_csv('{}_博文时间数据.csv'.format(name), encoding='utf-8-sig')

    new_df = time_df1['博文点赞']
    x_data = [str(x) for x in new_df.index]
    y_data = [int(x) for x in new_df.values]

    new_df1 = time_df1['博文评论']
    x_data1 = [str(x) for x in new_df1.index]
    y_data1 = [int(x) for x in new_df1.values]

    new_df2 = time_df1['博文转发']
    x_data2 = [str(x) for x in new_df2.index]
    y_data2 = [int(x) for x in new_df2.values]

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(15, 12), dpi=500)

    plt.plot(x_data, y_data, color='#b82410', label='点赞')
    plt.plot(x_data1, y_data1, '^--', color='#2614e8', label='评论')
    plt.plot(x_data2, y_data2, 'go--', label='转发')
    plt.legend()
    plt.title('{}_博文影响力趋势'.format(name))
    plt.xlabel('Day')
    plt.ylabel('values')
    plt.grid()
    plt.xticks(rotation=45)
    plt.savefig('{}_博文影响力趋势.png'.format(name))
    plt.show()


def area_influence_analyze2(name):
    df1 = pd.read_csv('一级评论数据.csv')
    central_area = ['河南', '湖北', '湖南', '江西', '安徽', '山西']
    west_area = ['内蒙古', '新疆', '宁夏', '甘肃', '青海', '陕西', '四川', '重庆', '贵州', '云南', '西藏']
    east_area = ['北京', '天津', '河北', '山东', '江苏', '上海', '浙江', '福建', '广东', '海南', '广西', '香港', '澳门', '台湾', '黑龙江', '辽宁', '吉林']

    def area(x):
        x1 = str(x)
        if x1 in central_area:
            return "中部地区"
        elif x1 in west_area:
            return "西部地区"
        elif x1 in east_area:
            return "东部地区"
        else:
            return "其他地区"

    df1['评论ip'] = df1['评论ip'].apply(area)

    df1 = df1[df1['评论ip'] == name]

    df1['评论时间'] = pd.to_datetime(df1['评论时间'])
    df1.index = df1['评论时间']
    time_df1 = df1['评论点赞'].resample('M').sum()

    time_df1.to_csv('{}_一级评论时间数据.csv'.format(name),encoding='utf-8-sig')

    x_data = [str(x).split(" ")[0] for x in time_df1.index]
    y_data = [int(x) for x in time_df1.values]

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(15, 12), dpi=500)

    plt.plot(x_data, y_data, color='#b82410', label='点赞')
    plt.legend()
    plt.title('{}_一级评论影响力趋势'.format(name))
    plt.xlabel('Month')
    plt.ylabel('values')
    plt.grid()
    plt.xticks(rotation=45)
    plt.savefig('{}_一级评论影响力趋势.png'.format(name))
    plt.show()


def influence_analyze2():
    df1 = pd.read_csv('一级评论数据.csv')
    df1['评论时间'] = pd.to_datetime(df1['评论时间'])
    df1.index = df1['评论时间']
    time_df1 = df1['评论点赞'].resample('M').sum()

    time_df1.to_csv('一级评论时间数据.csv',encoding='utf-8-sig')

    x_data = [str(x).split(" ")[0] for x in time_df1.index]
    y_data = [int(x) for x in time_df1.values]

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(15, 12), dpi=500)

    plt.plot(x_data, y_data, color='#b82410', label='点赞')
    plt.legend()
    plt.title('一级评论影响力趋势')
    plt.xlabel('Month')
    plt.ylabel('values')
    plt.grid()
    plt.xticks(rotation=45)
    plt.savefig('一级评论影响力趋势.png')
    plt.show()

def influence_analyze3():
    df1 = pd.read_csv('二级评论数据.csv')
    df1['二级评论时间'] = pd.to_datetime(df1['二级评论时间'])
    df1.index = df1['二级评论时间']
    time_df1 = df1['二级评论点赞'].resample('M').sum()

    time_df1.to_csv('二级评论时间数据.csv', encoding='utf-8-sig')

    x_data = [str(x).split(" ")[0] for x in time_df1.index]
    y_data = [int(x) for x in time_df1.values]

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(15, 12), dpi=500)

    plt.plot(x_data, y_data, color='#b82410', label='点赞')
    plt.legend()
    plt.title('二级评论影响力趋势')
    plt.xlabel('Month')
    plt.ylabel('values')
    plt.grid()
    plt.xticks(rotation=45)
    plt.savefig('二级评论影响力趋势.png')
    plt.show()


def area_influence_analyze3(name):
    df1 = pd.read_csv('二级评论数据.csv')
    central_area = ['河南', '湖北', '湖南', '江西', '安徽', '山西']
    west_area = ['内蒙古', '新疆', '宁夏', '甘肃', '青海', '陕西', '四川', '重庆', '贵州', '云南', '西藏']
    east_area = ['北京', '天津', '河北', '山东', '江苏', '上海', '浙江', '福建', '广东', '海南', '广西', '香港', '澳门', '台湾', '黑龙江', '辽宁', '吉林']

    def area(x):
        x1 = str(x)
        if x1 in central_area:
            return "中部地区"
        elif x1 in west_area:
            return "西部地区"
        elif x1 in east_area:
            return "东部地区"
        else:
            return "其他地区"

    df1['二级评论ip'] = df1['二级评论ip'].apply(area)

    df1 = df1[df1['二级评论ip'] == name]

    df1['二级评论时间'] = pd.to_datetime(df1['二级评论时间'])
    df1.index = df1['二级评论时间']
    time_df1 = df1['二级评论点赞'].resample('M').sum()

    time_df1.to_csv('{}_二级评论时间数据.csv'.format(name), encoding='utf-8-sig')

    x_data = [str(x).split(" ")[0] for x in time_df1.index]
    y_data = [int(x) for x in time_df1.values]

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(15, 12), dpi=500)

    plt.plot(x_data, y_data, color='#b82410', label='点赞')
    plt.legend()
    plt.title('{}_二级评论影响力趋势'.format(name))
    plt.xlabel('Month')
    plt.ylabel('values')
    plt.grid()
    plt.xticks(rotation=45)
    plt.savefig('{}_二级评论影响力趋势.png'.format(name))
    plt.show()


def lda(data,name):
    train = []
    for line in data:
        line = [word.strip(' ') for word in line.split(' ') if len(word) >= 2]
        train.append(line)

    # 构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    # 困惑度模块
    x_data = []
    y_data = []
    z_data = []
    for i in tqdm(range(2, 15)):
        x_data.append(i)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=i)
        # 困惑度计算
        perplexity = lda_model.log_perplexity(corpus)
        y_data.append(perplexity)
        # 一致性计算
        coherence_model_lda = CoherenceModel(model=lda_model, texts=train, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model_lda.get_coherence()
        z_data.append(coherence)

    # 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 绘制困惑度折线图
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x_data, y_data, marker="o")
    plt.title("perplexity_values")
    plt.xlabel('num topics')
    plt.ylabel('perplexity score')
    #绘制一致性的折线图
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x_data, z_data, marker="o")
    plt.title("coherence_values")
    plt.xlabel("num topics")
    plt.ylabel("coherence score")

    plt.savefig('{}_困惑度和一致性.png'.format(name))
    #将上面获取的数据进行保存
    df5 = pd.DataFrame()
    df5['主题数'] = x_data
    df5['困惑度'] = y_data
    df5['一致性'] = z_data
    df5.to_csv('{}_困惑度和一致性.csv'.format(name),encoding='utf-8-sig',index=False)

    optimal_z = max(z_data)
    optimal_z_index = z_data.index(optimal_z)
    best_topic_number = x_data[optimal_z_index]
    num_topics = best_topic_number
    # LDA可视化模块
    # 构建整体参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111,
                                          iterations=400)
    # 读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    # 把数据进行可视化处理
    pyLDAvis.save_html(data1, '{}_lda.html'.format(name))


if __name__ == '__main__':
    # user_analyze()
    # map_analyze()
    # emotion_analyze1()
    # emotion_analyze2()
    # emotion_analyze3()
    # list_area = ['东部地区', '西部地区', '中部地区']
    # for l in list_area:
    #     emotion_analyze4(l)
    # influence_analyze1()
    # influence_analyze2()
    # influence_analyze3()
    #
    # list_area = ['东部地区', '西部地区', '中部地区']
    # for l in list_area:
    #     area_influence_analyze1(l)
    #
    # list_area = ['东部地区', '西部地区', '中部地区']
    # for l in list_area:
    #     area_influence_analyze2(l)
    #
    # list_area = ['东部地区', '西部地区', '中部地区']
    # for l in list_area:
    #     area_influence_analyze3(l)

    df1 = pd.read_csv('博文数据.csv')
    df2 = pd.read_csv('一级评论数据.csv')
    df3 = pd.read_csv('二级评论数据.csv')

    # list_text1 = []
    # for d in df1['评论分词']:
    #     list_text1.append(d)
    # for d in df2['评论分词']:
    #     list_text1.append(d)
    # for d in df3['评论分词']:
    #     list_text1.append(d)
    #
    # lda(list_text1,'整体')
    #
    # list_sex = ['男','女']
    # for l in list_sex:
    #     list_text2 = []
    #     df22 = df2[df2['评论性别'] == l]
    #     df33 = df3[df3['二级评论性别'] == l]
    #
    #     for d in df22['评论分词']:
    #         list_text2.append(d)
    #     for d in df33['评论分词']:
    #         list_text2.append(d)
    #
    #     lda(list_text2, l)


    central_area = ['河南', '湖北', '湖南', '江西', '安徽', '山西']
    west_area = ['内蒙古', '新疆', '宁夏', '甘肃', '青海', '陕西', '四川', '重庆', '贵州', '云南', '西藏']
    east_area = ['北京', '天津', '河北', '山东', '江苏', '上海', '浙江', '福建', '广东', '海南', '广西', '香港', '澳门', '台湾', '黑龙江', '辽宁', '吉林']

    def area(x):
        x1 = str(x)
        if x1 in central_area:
            return "中部地区"
        elif x1 in west_area:
            return "西部地区"
        elif x1 in east_area:
            return "东部地区"
        else:
            return "其他地区"


    df1['发博用户ip'] = df1['发博用户ip'].apply(area)
    df2['评论ip'] = df2['评论ip'].apply(area)
    df3['二级评论ip'] = df3['二级评论ip'].apply(area)
    list_area = ['东部地区', '西部地区', '中部地区']
    for l in list_area:
        list_text3 = []
        df11 = df1[df1['发博用户ip'] == l]
        df22 = df2[df2['评论ip'] == l]
        df33 = df3[df3['二级评论ip'] == l]

        for d in df11['评论分词']:
            list_text3.append(d)
        for d in df22['评论分词']:
            list_text3.append(d)
        for d in df33['评论分词']:
            list_text3.append(d)

        lda(list_text3, l)
