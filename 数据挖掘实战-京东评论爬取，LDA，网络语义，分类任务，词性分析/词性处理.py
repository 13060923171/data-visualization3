import jieba
import pandas as pd
import jieba.posseg as posseg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from IPython.display import Image
import stylecloud

def main1(x):
    df1 = df[df['class'] == x]
    counts = {}
    for i in df1['物流信息']:
        res = posseg.cut(i)
        for word, flag in res:
            if flag == 'Ng' or flag == 'n' or flag == 'nr' or flag == 'ns' or flag == 'nt' or flag == 'nz':
                counts[word] = counts.get(word, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x:x[1],reverse=True)
    x_data = []
    y_data = []

    for key, values in ls:
        if values > 5:
            x_data.append(key)
            y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./词性数据/{}_名词.csv'.format(x), encoding="utf-8-sig")


def main2(x):
    df1 = df[df['class'] == x]
    corpus = []
    for i in df1['分词']:
        corpus.append(i.strip())

        # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names_out()

    # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    data = {'word': word,
            'tfidf': weight.sum(axis=0).tolist()}
    df2 = pd.DataFrame(data)
    df2['tfidf'] = df2['tfidf'].astype('float64')
    df2 = df2.sort_values(by=['tfidf'], ascending=False)

    x_data = list(df2['word'])[:20]
    y_data = list(df2['tfidf'])[:20]
    x_data.reverse()
    y_data.reverse()
    plt.figure(figsize=(12, 9), dpi=300)
    plt.barh(x_data, y_data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("TF-IDF TOP20_{}".format(x))
    plt.xlabel("数值")
    plt.savefig('./词性数据/{}_TFIDF.png'.format(x))
    plt.show()
    df2.to_csv('./词性数据/{}_tfidf.csv'.format(x), encoding='utf-8-sig', index=False)


def main3(x):
    df1 = df[df['class'] == x]
    str1 = ''
    counts = {}
    for i in df1['物流信息']:
        res = posseg.cut(i)
        for word, flag in res:
            if flag == 'Ag' or flag == 'a' or flag == 'ad' or flag == 'an':
                counts[word] = counts.get(word, 0) + 1
                str1 += word + ' '

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls:
        if values > 5:
            x_data.append(key)
            y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./词性数据/{}_形容词.csv'.format(x), encoding="utf-8-sig")

    str1 = str1.strip(' ')
    stylecloud.gen_stylecloud(text=str1, max_words=100,
                              collocations=False,
                              background_color="#B3B6B7",
                              font_path='simhei.ttf',
                              icon_name='fas fa-tree',
                              size=500,
                              palette='matplotlib.Inferno_9',
                              output_name='./词性数据/{}_词云图.png'.format(x))
    Image(filename='./词性数据/{}_词云图.png'.format(x))


def main4():
    str1 = ''
    for i in df['物流信息']:
        res = posseg.cut(i)
        for word, flag in res:
            if flag == 'Ag' or flag == 'a' or flag == 'ad' or flag == 'an':
                str1 += word + ' '
    str1 = str1.strip(' ')
    stylecloud.gen_stylecloud(text=str1, max_words=100,
                              collocations=False,
                              background_color="#B3B6B7",
                              font_path='simhei.ttf',
                              icon_name='fas fa-heart',
                              size=500,
                              palette='matplotlib.Inferno_9',
                              output_name='./词性数据/总体_词云图.png'.format(x))
    Image(filename='./词性数据/总体_词云图.png'.format(x))


if __name__ == '__main__':
    df = pd.read_csv('new_data.csv')
    new_df = df['class'].value_counts()
    x_data1 = [x for x in new_df.index]
    for x in x_data1:
        main1(x)
        main2(x)
        main3(x)
    main4()