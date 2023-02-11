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
from gensim import corpora, models
import itertools
import pyLDAvis
import pyLDAvis.gensim
import gensim
from tqdm import tqdm
import os
from gensim.models import LdaModel

# 这里使用了百度开源的成熟NLP模型来预测情感倾向
import paddlehub as hub
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def data_processing():
    df1 = pd.read_csv('data1.csv',encoding='gbk')
    df2 = pd.read_csv('data2.csv',encoding='gbk')
    df3 = pd.read_csv('data3.csv', encoding='gbk')
    data = pd.concat([df1,df2,df3],axis=0)
    data = data.drop_duplicates(subset=['提问内容'],keep='first')
    data = data.rename(columns={'问题领域':'提问类型','提问类型':'问题领域'})
    data.to_csv('all_data.csv',encoding='utf-8-sig',index=False)


#中文判断函数#
def wordclound_fx(name=None):
    df = pd.read_csv('all_data.csv')
    df = df.dropna(subset=[name], axis=0)
    content = df[name].drop_duplicates(keep='first')
    content = content.dropna(how='any')
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    def get_cut_words(content_series):
        # 读入停用词表
        stop_words = ['领导','期间','您好','尊敬','疫情','新冠','肺炎','感谢您','留言']

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        # 分词
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    text3 = get_cut_words(content_series=content)
    stylecloud.gen_stylecloud(text=' '.join(text3), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-circle',
                              # icon_name='fas fa-star',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./data/{}-词云图.png'.format(name))
    Image(filename='./data/{}-词云图.png'.format(name))

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
    df1.to_csv('./data/{}TOP200_高频词.csv'.format(name), encoding="utf-8-sig")


def snownlp():
    df = pd.read_csv('all_data.csv')
    df = df.dropna(subset=['提问内容','官方答复'], axis=0)

    senta = hub.Module(name="senta_bilstm")

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

    def get_cut_words(str1):
        # 读入停用词表
        stop_words = ['领导','期间','您好','尊敬','疫情','新冠','肺炎','感谢您','留言']

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        # 分词
        word_num = jieba.lcut(str(str1), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

        word_num_selected = " ".join(word_num_selected)

        if len(word_num_selected) != 0:
            return word_num_selected
        else:
            return np.NAN

    def txt_len(x):
        x1 = str(x)
        return len(x1)

    def date_time(x):
        x1 = str(x)
        x1 = x1.replace("?"," ")
        return x1

    def time_qx(x):
        x1 = str(x)
        x1 = x1.split("days")
        x1 = x1[0]
        return int(x1)

    df['提问'] = df['提问内容'].apply(emjio_tihuan)
    df = df.dropna(subset=['提问'], axis=0)
    df['提问'] = df['提问'].apply(get_cut_words)
    df = df.dropna(subset=['提问'], axis=0)

    df['官方'] = df['官方答复'].apply(emjio_tihuan)
    df = df.dropna(subset=['官方'], axis=0)
    df['官方'] = df['官方'].apply(get_cut_words)
    df = df.dropna(subset=['官方'], axis=0)
    df['提问时间'] = df['提问时间'].apply(date_time)
    df['答复时间'] = df['答复时间'].apply(date_time)
    df['相差间隔'] = pd.to_datetime(df['答复时间'])-pd.to_datetime(df['提问时间'])
    df['相差间隔'] = df['相差间隔'].apply(time_qx)

    texts = df['提问'].tolist()
    input_data = {'text': texts}
    res = senta.sentiment_classify(data=input_data)
    df['提问内容_长度'] = df['提问内容'].apply(txt_len)
    df['提问内容_positive_probs'] = [x['positive_probs'] for x in res]
    df['提问内容_negative_probs'] = [x['negative_probs'] for x in res]

    texts1 = df['官方'].tolist()
    input_data1 = {'text': texts1}
    res1 = senta.sentiment_classify(data=input_data1)
    df['官方答复_长度'] = df['官方答复'].apply(txt_len)
    df['官方答复_positive_probs'] = [x['positive_probs'] for x in res1]
    df['官方答复_negative_probs'] = [x['negative_probs'] for x in res1]

    df.to_csv('./data/nlp_all_data.csv',encoding='utf-8-sig',index=False)


def nlp_picture():
    df = pd.read_csv('./data/nlp_all_data.csv',parse_dates=['提问时间'], index_col="提问时间")
    df5 = pd.read_csv('./data/nlp_all_data.csv', parse_dates=['答复时间'], index_col="答复时间")

    df1 = df['提问内容_positive_probs'].resample('M').mean()
    df3 = df5['官方答复_positive_probs'].resample('M').mean()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(20,9),dpi=300)
    plt.plot(df1,'^--',color='#b82410', label='提问内容')
    plt.plot(df3,'o--',color='#2614e8', label='官方答复')
    plt.title('公民诉求VS政府回应-正面情感月均值趋势')
    plt.xlabel('月份')
    plt.ylabel('均值')
    plt.grid()
    plt.legend()
    plt.savefig('./data/提问-情感趋势.png')
    plt.show()

    df2 = df['提问内容_negative_probs'].resample('M').mean()
    df4 = df5['官方答复_negative_probs'].resample('M').mean()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(20, 9), dpi=300)
    plt.plot(df2, '^--', color='#b82410', label='提问内容')
    plt.plot(df4, 'o--', color='#2614e8', label='官方答复')
    plt.title('公民诉求VS政府回应-负面情感月均值趋势')
    plt.xlabel('月份')
    plt.ylabel('均值')
    plt.grid()
    plt.legend()
    plt.savefig('./data/答复-情感趋势.png')
    plt.show()


def nlp_kmeans(name=None):
    df = pd.read_csv('all_data.csv')
    df = df.dropna(subset=[name], axis=0)
    content = df[name].drop_duplicates(keep='first')
    content = content.dropna(how='any')

    stop_words = ['领导','期间','您好','尊敬','疫情','新冠','肺炎','感谢您','留言']

    with open("stopwords_cn.txt", 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    f = open('./data/{}-fenci.txt'.format(name), 'w', encoding='utf-8-sig')
    for line in content:
        line = line.strip('\n')
        # 停用词过滤
        line = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", line)
        seg_list = jieba.cut(line, cut_all=False)
        cut_words = (" ".join(seg_list))

        # 计算关键词
        all_words = cut_words.split()
        c = Counter()
        for x in all_words:
            if len(x) >= 2 and x != '\r\n' and x != '\n':
                if is_all_chinese(x) == True and x not in stop_words:
                    c[x] += 1
        # Top30
        output = ""
        # print('\n词频统计结果：')
        for (k, v) in c.most_common(30):
            # print("%s:%d"%(k,v))
            output += k + " "

        f.write(output + "\n")
    else:
        f.close()

    corpus = []

    # 读取预料 一行预料为一个文档
    for line in open('./data/{}-fenci.txt'.format(name), 'r', encoding='utf-8-sig').readlines():
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

    n_clusters = 2
    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))
    # 第二步 聚类Kmeans
    print('Start Kmeans:')

    clf = KMeans(n_clusters=n_clusters)
    pre = clf.fit_predict(weight)

    # 中心点
    print(clf.cluster_centers_)
    print(clf.inertia_)

    #图形输出 降维

    pca = PCA(n_components=n_clusters)  # 输出两维
    newData = pca.fit_transform(weight)  # 载入N维

    x = [n[0] for n in newData]
    y = [n[1] for n in newData]
    plt.figure(figsize=(12, 9), dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(x, y, c=pre, s=100)
    plt.title("{}-情感聚类图".format(name))
    plt.savefig('./data/{}-情感聚类图.png'.format(name))
    plt.show()

#LDA建模
def lda(name=None):
    df = pd.read_csv('all_data.csv')
    df = df.dropna(subset=[name], axis=0)
    content = df[name].drop_duplicates(keep='first')
    content = content.dropna(how='any')

    fr = open('./data/{}-fenci.txt'.format(name), 'r', encoding='utf-8-sig')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ') if len(word) >= 2]
        train.append(line)

    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    # # 构造主题数寻优函数
    # def cos(vector1, vector2):  # 余弦相似度函数
    #     dot_product = 0.0
    #     normA = 0.0
    #     normB = 0.0
    #     for a, b in zip(vector1, vector2):
    #         dot_product += a * b
    #         normA += a ** 2
    #         normB += b ** 2
    #     if normA == 0.0 or normB == 0.0:
    #         return (None)
    #     else:
    #         return (dot_product / ((normA * normB) ** 0.5))
    #
    #     # 主题数寻优
    #
    # def lda_k(x_corpus, x_dict):
    #     # 初始化平均余弦相似度
    #     mean_similarity = []
    #     mean_similarity.append(1)
    #
    #     # 循环生成主题并计算主题间相似度
    #     for i in np.arange(2, 11):
    #         lda = models.LdaModel(x_corpus, num_topics=i, id2word=x_dict)  # LDA模型训练
    #         for j in np.arange(i):
    #             term = lda.show_topics(num_words=30)
    #
    #         # 提取各主题词
    #         top_word = []
    #         for k in np.arange(i):
    #
    #             top_word.append([''.join(re.findall('"(.*)"', i)) \
    #                              for i in term[k][1].split('+')])  # 列出所有词
    #
    #         # 构造词频向量
    #         word = sum(top_word, [])  # 列出所有的词
    #         unique_word = set(word)  # 去除重复的词
    #
    #         # 构造主题词列表，行表示主题号，列表示各主题词
    #         mat = []
    #         for j in np.arange(i):
    #             top_w = top_word[j]
    #             mat.append(tuple([top_w.count(k) for k in unique_word]))
    #
    #         p = list(itertools.permutations(list(np.arange(i)), 2))
    #         l = len(p)
    #         top_similarity = [0]
    #         for w in np.arange(l):
    #             vector1 = mat[p[w][0]]
    #             vector2 = mat[p[w][1]]
    #             top_similarity.append(cos(vector1, vector2))
    #
    #         # 计算平均余弦相似度
    #         mean_similarity.append(sum(top_similarity) / l)
    #     return (mean_similarity)
    #
    # # 计算主题平均余弦相似度
    # word_k = lda_k(corpus, dictionary)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(10, 8), dpi=300)
    # plt.plot(word_k)
    # plt.title('{}-LDA评论主题数寻优'.format(name))
    # plt.xlabel('主题数')
    # plt.ylabel('平均余弦相似度')
    # plt.savefig('./data/{}-主题数寻优.png'.format(name))
    # plt.show()
    # 困惑度模块
    x_data = []
    y_data = []
    z_data = []
    for i in tqdm(range(2, 15)):
        x_data.append(i)
        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, random_state=111, iterations=400)
        # 困惑度计算
        _,perplexity = lda.log_perplexity(corpus)
        y_data.append(perplexity)
        # 一致性计算
        coherencemodel = models.CoherenceModel(model=lda, texts=train, dictionary=dictionary, coherence='c_v')
        z_data.append(coherencemodel.get_coherence())

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

    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x_data, z_data, marker="o")
    plt.title("coherence_values")
    plt.xlabel("num topics")
    plt.ylabel("coherence score")

    plt.savefig('./data/{}-困惑度和一致性.png'.format(name))
    plt.show()

    df5 = pd.DataFrame()
    df5['主题数'] = x_data
    df5['困惑度'] = y_data
    df5['一致性'] = z_data
    df5.to_csv('./data/{}-困惑度和一致性.csv'.format(name),encoding='utf-8-sig',index=False)
    num_topics = input('请输入主题数:')

    #LDA可视化模块
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(data1, './data/{}-lda.html'.format(name))

    #主题判断模块
    list3 = []
    list2 = []
    for i in lda.get_document_topics(corpus)[:]:
        listj = []
        list1 = []
        for j in i:
            list1.append(j)
            listj.append(j[1])
        list3.append(list1)
        bz = listj.index(max(listj))
        list2.append(i[bz][0])

    data = pd.DataFrame()
    data['内容'] = content
    data['主题概率'] = list3
    data['主题类型'] = list2

    data.to_csv('./data/{}-all_data.csv'.format(name),encoding='utf-8-sig',index=False)

    new_data = data['主题类型'].value_counts()
    new_data = new_data.sort_index(ascending=True)
    y_data1 = [y for y in new_data.values]

    #主题词模块
    word = lda.print_topics(num_words=20)
    topic = []
    quanzhong = []
    list_gailv = []
    list_gailv1 = []

    list_word = []


    for w in word:
        ci = str(w[1])
        c1 = re.compile('\*"(.*?)"')
        c2 = c1.findall(ci)
        list_word.append(c2)
        c3 = '、'.join(c2)

        c4 = re.compile(".*?(\d+).*?")
        c5 = c4.findall(ci)
        for c in c5[::1]:
            if c != "0":
                gailv = str(0) + '.' + str(c)
                list_gailv.append(gailv)
        list_gailv1.append(list_gailv)
        list_gailv = []
        zt = "Topic" + str(w[0])
        topic.append(zt)
        quanzhong.append(c3)

    df2 = pd.DataFrame()
    for j,k,l in zip(topic,list_gailv1,list_word):
        df2['{}-主题词'.format(j)] = l
        df2['{}-权重'.format(j)] = k
    df2.to_csv('./data/{}-主题词分布表.csv'.format(name), encoding='utf-8-sig', index=False)

    y_data2 = []
    for y in y_data1:
        number = float(y / sum(y_data1))
        y_data2.append(float('{:0.5}'.format(number)))

    df1 = pd.DataFrame()
    df1['所属主题'] = topic
    df1['文章数量'] = y_data1
    df1['特征词'] = quanzhong
    df1['主题强度'] = y_data2
    df1.to_csv('./data/{}-特征词.csv'.format(name),encoding='utf-8-sig',index=False)


if __name__ == '__main__':
    # 官方答复 提问内容
    # data_processing()
    # wordclound_fx(name='官方答复')
    snownlp()
    # nlp_picture()
    #
    # nlp_kmeans(name='官方答复')
    # lda(name='官方答复')
