import pandas as pd
import numpy as np
import re
import jieba
import pyLDAvis
import pyLDAvis.gensim
from tqdm import tqdm
import os
from gensim.models import LdaModel
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import matplotlib
from snownlp import SnowNLP
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split # 用于数据集划分
from sklearn.preprocessing import StandardScaler # 用于数据归一化
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import r2_score  # 用于计算R2 Score
from pyecharts import options as opts
from pyecharts.charts import Bar
import scikitplot as skplt

stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


def data_processing():
    df = pd.read_csv('processed_comments.csv', encoding='gbk')
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

    df['content1'] = df['content'].apply(emjio_tihuan)
    df = df.dropna(subset=['content1'], axis=0)
    df['content1'] = df['content1'].apply(yasuo)
    df['分词'] = df['content1'].apply(get_cut_words)
    df.to_csv('new_data.csv', encoding='utf-8-sig', index=False)


#LDA建模
def lda():
    df = pd.read_csv('new_data.csv')
    df = df.dropna(how='any',axis=0)
    train = []
    for line in df['分词']:
        line = [word.strip(' ') for word in line.split(' ') if len(word) >= 2]
        train.append(line)

    #构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    #
    # # 困惑度模块
    # x_data = []
    # y_data = []
    # z_data = []
    # for i in tqdm(range(2, 15)):
    #     x_data.append(i)
    #     lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=i)
    #     # 困惑度计算
    #     perplexity = lda_model.log_perplexity(corpus)
    #     y_data.append(perplexity)
    #     # 一致性计算
    #     coherence_model_lda = CoherenceModel(model=lda_model, texts=train, dictionary=dictionary, coherence='c_v')
    #     coherence = coherence_model_lda.get_coherence()
    #     z_data.append(coherence)
    #
    # # 绘制困惑度和一致性折线图
    # fig = plt.figure(figsize=(15, 5))
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False
    #
    # # 绘制困惑度折线图
    # ax1 = fig.add_subplot(1, 2, 1)
    # plt.plot(x_data, y_data, marker="o")
    # plt.title("perplexity_values")
    # plt.xlabel('num topics')
    # plt.ylabel('perplexity score')
    # #绘制一致性的折线图
    # ax2 = fig.add_subplot(1, 2, 2)
    # plt.plot(x_data, z_data, marker="o")
    # plt.title("coherence_values")
    # plt.xlabel("num topics")
    # plt.ylabel("coherence score")
    #
    # plt.savefig('./LDA主题/困惑度和一致性.png')
    # plt.show()
    # #将上面获取的数据进行保存
    # df5 = pd.DataFrame()
    # df5['主题数'] = x_data
    # df5['困惑度'] = y_data
    # df5['一致性'] = z_data
    # df5.to_csv('./LDA主题/困惑度和一致性.csv',encoding='utf-8-sig',index=False)
    num_topics = input('请输入主题数:')

    #LDA可视化模块
    #构建lda主题参数
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    #读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    #把数据进行可视化处理
    pyLDAvis.save_html(data1, './LDA主题/lda.html')

    #主题判断模块
    list3 = []
    list2 = []
    #这里进行lda主题判断
    for i in lda.get_document_topics(corpus)[:]:
        listj = []
        list1 = []
        for j in i:
            list1.append(j)
            listj.append(j[1])
        list3.append(list1)
        bz = listj.index(max(listj))
        list2.append(i[bz][0])

    # data = pd.DataFrame()
    # data['内容'] = df['分词']
    df['主题概率'] = list3
    df['主题类型'] = list2

    df.to_csv('lda_data.csv',encoding='utf-8-sig',index=False)


def snownlp_type():
    data = pd.read_csv('lda_data.csv')
    def main1(x):
        comment = x.strip()
        # 创建一个SnowNLP对象
        s = SnowNLP(comment)
        # 调用sentiments属性，得到情感倾向值
        sentiment = s.sentiments
        sentiment = float(sentiment)
        if sentiment <= 0.35:
            return '负面'
        else:
            return '非负'
    data['emotion_type'] = data['分词'].apply(main1)
    data.to_csv('data_情感分析.csv', encoding='utf-8-sig', index=False)


def learning():
    df = pd.read_csv('data_情感分析.csv', encoding='utf-8-sig')
    # new_df = df[df['主题类型'] == 0]

    # 创建LabelEncoder对象
    le = LabelEncoder()
    # 使用LabelEncoder对象对分类变量进行编码
    df['emotion_type'] = le.fit_transform(df['emotion_type'])
    features = ['主题类型','emotion_type']
    x = df[features]
    # 将分词后的中文文本转换为数字类型
    y = list(df['评价'])

    # # 对数据集进行划分，按照8:2的比例分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  #

    # 创建三种回归模型
    rf = RandomForestRegressor()  # 随机森林回归器
    xgb = XGBRegressor()  # XGboost回归器
    lgb = LGBMRegressor()  # LightGBM回归器

    ###########2.回归部分##########
    def regression_method(model,name):
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        result = model.predict(X_test)
        ResidualSquare = (result - y_test) ** 2  # 计算残差平方
        RSS = sum(ResidualSquare)  # 计算残差平方和
        MSE = np.mean(ResidualSquare)  # 计算均方差
        num_regress = len(result)  # 回归样本个数
        print(f'n={num_regress}')
        print(f'R^2={score}')
        print(f'MSE={MSE}')
        print(f'RSS={RSS}')
        ############绘制折线图##########
        plt.figure()
        plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
        plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
        plt.title('{} R^2: {}'.format(name,score))
        plt.legend()  # 将样例显示出来
        plt.savefig('{}.png'.format(name))
        plt.show()

        data = pd.DataFrame()
        model = ['回归样本个数','R2','均方差','残差平方和']
        r2 = [num_regress, score, MSE,RSS]
        data['指标'] = model
        data['分值'] = r2
        data.to_csv('{}_model.csv'.format(name), encoding='utf-8-sig')
        return result

    ##########3.绘制验证散点图########
    def scatter_plot(TureValues, PredictValues,name):
        # 设置参考的1：1虚线参数
        xxx = [-0.5, 1.5]
        yyy = [-0.5, 1.5]
        # 绘图
        plt.figure()
        plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
        plt.scatter(TureValues, PredictValues, s=20, c='r', edgecolors='k', marker='o',
                    alpha=0.8)  # 绘制散点图，横轴是真实值，竖轴是预测值
        plt.xlim((0, 1))  # 设置坐标轴范围
        plt.ylim((0, 1))
        plt.title('{}_ScatterPlot'.format(name))
        plt.savefig('{}_ScatterPlot.png'.format(name))
        plt.show()

    y_pred = regression_method(rf,'rf')  # 括号内填上方法，并获取预测值
    scatter_plot(y_test, y_pred,'rf')  # 生成散点图

    y_pred = regression_method(xgb,'xgb')  # 括号内填上方法，并获取预测值
    scatter_plot(y_test, y_pred,'xgb')  # 生成散点图

    y_pred = regression_method(lgb,'lgb')  # 括号内填上方法，并获取预测值
    scatter_plot(y_test, y_pred,'lgb')  # 生成散点图


def type1_bar():
    df = pd.read_csv('data_情感分析.csv', encoding='utf-8-sig')
    def main(x):
        new_df = df[df['主题类型'] == x]
        new_df1 = new_df['评价'].value_counts()
        new_df1 = new_df1.sort_index()
        x_data = [str(x) for x in new_df1.index]
        y_data = [int(y) for y in new_df1.values]
        return x_data,y_data
    X_DATA = []
    Y_DATA = []
    for i in range(0,3):
        x_data,y_data = main(i)
        X_DATA.append(x_data)
        Y_DATA.append(y_data)
    c = (
        Bar()
            .add_xaxis(X_DATA[0])
            .add_yaxis("主题-0", Y_DATA[0])
            .add_yaxis("主题-1", Y_DATA[1])
            .add_yaxis("主题-2", Y_DATA[2])
            .set_global_opts(
            title_opts=opts.TitleOpts(title="主题评价概况"),
        )
            .render("主题评价概况.html")
    )

def type_bar():
    df = pd.read_csv('data_情感分析.csv', encoding='utf-8-sig')
    def main(x):
        new_df = df[df['主题类型'] == x]
        new_df1 = new_df['emotion_type'].value_counts()
        new_df1 = new_df1.sort_index()
        x_data = [x for x in new_df1.index]
        y_data = [int(y) for y in new_df1.values]
        return x_data,y_data
    X_DATA = []
    Y_DATA = []
    for i in range(0,3):
        x_data,y_data = main(i)
        X_DATA.append(x_data)
        Y_DATA.append(y_data)
    c = (
        Bar()
            .add_xaxis(X_DATA[0])
            .add_yaxis("主题-0", Y_DATA[0])
            .add_yaxis("主题-1", Y_DATA[1])
            .add_yaxis("主题-2", Y_DATA[2])
            .set_global_opts(
            title_opts=opts.TitleOpts(title="主题情感分布"),
        )
            .render("主题情感分布.html")
    )

if __name__ == '__main__':
    # data_processing()
    # lda()
    # snownlp_type()
    learning()
    # type_bar()
    # type1_bar()