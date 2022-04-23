import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from snownlp import SnowNLP
import re
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import Image
import stylecloud
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn



def snownlp_fx():
    df = pd.read_excel('三星最终数据.xlsx')
    content = df['评论'].drop_duplicates(keep='first')
    content = content.dropna(how='any')

    def emotion_scroe(x):
        text = re.sub(r'(?:回复)?(?://)?@[\w\u2E80-\u9FFF]+:?|\[\w+\]', ',', x)
        text = re.sub(r'\n', '', text)
        score = SnowNLP(text)
        fenshu = score.sentiments
        return fenshu

    df1 = pd.DataFrame()
    df1['content'] = content
    df1['emotion_scroe'] = df1['content'].apply(emotion_scroe)
    df1.to_csv('./三星-data/三星-snownlp情感分析.csv',encoding='utf-8-sig',index=False)


def wordclound_fx():
    df = pd.read_csv('./三星-data/三星-snownlp情感分析.csv')
    df1 = df[df['emotion_scroe'] >= 0.5]
    df2 = df[df['emotion_scroe'] < 0.5]

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    def get_cut_words(content_series):
        # 读入停用词表
        stop_words = []

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())
        # 分词
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    text1 = get_cut_words(content_series=df1['content'])
    stylecloud.gen_stylecloud(text=' '.join(text1), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-circle',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./三星-data/正向词云图.png')
    Image(filename='./三星-data/正向词云图.png')

    text2 = get_cut_words(content_series=df2['content'])
    stylecloud.gen_stylecloud(text=' '.join(text2), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-anchor',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./三星-data/负向词云图.png')
    Image(filename='./三星-data/负向词云图.png')

    text3 = get_cut_words(content_series=df['content'])
    stylecloud.gen_stylecloud(text=' '.join(text3), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-atom',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./三星-data/总体词云图.png')
    Image(filename='./三星-data/总体词云图.png')

    counts = {}
    for t in text3:
        counts[t] = counts.get(t, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[:200]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./三星-data/高频词.csv', encoding="utf-8-sig")


def lda_tfidf():
    df = pd.read_csv('./三星-data/三星-snownlp情感分析.csv')

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    jieba.analyse.set_stop_words('stopwords_cn.txt')
    f = open('./三星-data/三星-class-fenci.txt', 'w', encoding='utf-8')
    for line in df['content']:
        line = line.strip('\n')
        # 停用词过滤
        line = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", line)
        seg_list = jieba.cut(line, cut_all=False)
        cut_words = (" ".join(seg_list))

        # 计算关键词
        all_words = cut_words.split()
        c = Counter()
        for x in all_words:
            if len(x) > 1 and x != '\r\n':
                if is_all_chinese(x) == True:
                    c[x] += 1
        # Top50
        output = ""
        # print('\n词频统计结果：')
        for (k, v) in c.most_common():
            # print("%s:%d"%(k,v))
            output += k + " "

        f.write(output + "\n")
    else:
        f.close()

    corpus = []

    # 读取预料 一行预料为一个文档
    for line in open('./三星-data/三星-class-fenci.txt', 'r', encoding='utf-8').readlines():
        corpus.append(line.strip())

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()

    # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    data = {'word': word,
            'tfidf': weight.sum(axis=0).tolist()}
    df2 = pd.DataFrame(data)
    df2['tfidf'] = df2['tfidf'].astype('float64')
    df2 = df2.sort_values(by=['tfidf'], ascending=False)
    df2.to_csv('./三星-data/三星-tfidf.csv', encoding='utf-8-sig',index=False)

    # 设置特征数
    n_features = 2000

    tf_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                    max_features=n_features,
                                    max_df=0.99,
                                    min_df=0.002)  # 去除文档内出现几率过大或过小的词汇

    tf = tf_vectorizer.fit_transform(corpus)


    # 设置主题数
    n_topics = 5

    lda = LatentDirichletAllocation(n_components=n_topics,
                                    max_iter=100,
                                    learning_method='online',
                                    learning_offset=50,
                                    random_state=0)
    lda.fit(tf)

    # 显示主题数 model.topic_word_
    print(lda.components_)
    # 几个主题就是几行 多少个关键词就是几列
    print(lda.components_.shape)

    # 计算困惑度
    print(u'困惑度：')
    print(lda.perplexity(tf, sub_sampling=False))

    # 主题-关键词分布
    def print_top_words(model, tf_feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):  # lda.component相当于model.topic_word_
            print('Topic #%d:' % topic_idx)
            print(' '.join([tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print("")

    # 定义好函数之后 暂定每个主题输出前20个关键词
    n_top_words = 20
    tf_feature_names = tf_vectorizer.get_feature_names()
    # 调用函数
    print_top_words(lda, tf_feature_names, n_top_words)

    data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

    pyLDAvis.save_html(data, './三星-data/三星-lda.html')

if __name__ == '__main__':
    snownlp_fx()
    wordclound_fx()
    lda_tfidf()