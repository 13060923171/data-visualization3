from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


def tf_idf(df,name):
    # corpus = []
    # for i in df['fenci']:
    #     corpus.append(i.strip())
    #
    #     # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    # vectorizer = CountVectorizer()
    #
    # # 该类会统计每个词语的tf-idf权值
    # transformer = TfidfTransformer()
    #
    # # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    # tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # # 获取词袋模型中的所有词语
    # word = vectorizer.get_feature_names_out()
    #
    # # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
    # weight = tfidf.toarray()
    #
    # data = {'word': word,
    #         'tfidf': weight.sum(axis=0).tolist()}
    #
    # df2 = pd.DataFrame(data)
    # df2['tfidf'] = df2['tfidf'].astype('float64')
    # df2 = df2.sort_values(by=['tfidf'],ascending=False)
    # df2.to_csv('{}-TF-IDF相关数据.csv'.format(name),encoding='utf-8-sig',index=False)
    #
    # df3 = df2.iloc[:30]
    df3 = pd.read_excel(f'{name}-TF-IDF相关数据.xlsx').iloc[:30]
    x_data = list(df3['word'])
    y_data = list(df3['tfidf'])
    x_data.reverse()
    y_data.reverse()
    plt.figure(figsize=(12, 9))
    plt.barh(x_data, y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.title("tf-idf 权重最高的top30词汇")
    plt.xlabel("权重")
    plt.savefig('{}-tf-idf top30.png'.format(name))


if __name__ == '__main__':
    df = pd.read_csv('new_data.csv')
    list_name = ['positive','negative']
    for n in list_name:
        df1 = df[df['label'] == n]
        tf_idf(df1,n)
    #
    # tf_idf(df,'整体')