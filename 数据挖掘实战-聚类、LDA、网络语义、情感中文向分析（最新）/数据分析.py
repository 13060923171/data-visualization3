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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim import corpora, models
import itertools
import pyLDAvis
import pyLDAvis.gensim
import gensim
from tqdm import tqdm
import os
import paddlehub as hub
import codecs
import networkx as nx
from scipy.sparse import coo_matrix

stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())

def snownlp_fx():
    df = pd.read_excel('./E9新闻/重庆大学.xlsx')
    data = df.drop_duplicates(keep='first')
    if not os.path.exists("./重庆大学"):
        os.mkdir("./重庆大学")

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

    def get_cut_words(content_series):
        # 读入停用词表
        # 分词
        word_num = jieba.lcut(content_series, cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

        return ' '.join(word_num_selected)

    data['新闻内容'] = data['新闻内容'].apply(emjio_tihuan)
    data = data.dropna(subset=['新闻内容'], axis=0)
    data['分词'] = data['新闻内容'].apply(get_cut_words)
    senta = hub.Module(name="senta_bilstm")
    texts = data['分词'].tolist()
    input_data = {'text': texts}
    res = senta.sentiment_classify(data=input_data)
    data['情感分值'] = [x['positive_probs'] for x in res]
    data = data.dropna(how='any', axis=0)
    data.to_csv('./重庆大学/重庆大学_情感分析.csv', encoding='utf-8-sig', index=False)


def wordclound_fx():
    data = pd.read_csv('./重庆大学/重庆大学_情感分析.csv')
    text1 = [x for x in data['分词']]
    stylecloud.gen_stylecloud(text=' '.join(map(str,text1)), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-circle',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='./重庆大学/高频-词云图.png')
    Image(filename='./重庆大学/高频-词云图.png')

    counts = {}
    for t in text1:
        counts[t] = counts.get(t, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[:100]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./重庆大学/总高频词TOP100.csv', encoding="utf-8-sig")


def emotion_score():
    data = pd.read_csv('./重庆大学/重庆大学_情感分析.csv')
    def main1(x):
        x1 = float(x)
        if x1 <= 0.45:
            return 'negative'
        elif 0.45 < x1 < 0.55:
            return 'neutral'
        else:
            return 'positive'
    data['emotion_type'] = data['情感分值'].apply(main1)
    new_df = data['emotion_type'].value_counts()
    new_df.sort_index(inplace=True)
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(9, 6),dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, shadow=True, startangle=0, autopct='%1.2f%%',
            wedgeprops={'edgecolor': 'black'})
    plt.savefig('./重庆大学/情感分布.png')

    def main2(x):
        df = data[data['emotion_type'] == str(x)]
        text1 = [x for x in df['分词']]
        text1 =' '.join(map(str,text1))
        text1 = text1.split(' ')
        counts = {}
        for t in text1:
            counts[t] = counts.get(t, 0) + 1

        ls = list(counts.items())
        ls.sort(key=lambda x: x[1], reverse=True)
        x_data = []
        y_data = []

        for key, values in ls:
            x_data.append(key)
            y_data.append(values)

        senta = hub.Module(name="senta_bilstm")
        input_data = {'text': x_data}
        res = senta.sentiment_classify(data=input_data)
        z_data = [x['positive_probs'] for x in res]
        df1 = pd.DataFrame()
        df1['{}-word'.format(x)] = x_data
        df1['{}-counts'.format(x)] = y_data
        df1['{}-score'.format(x)] = z_data
        if x == 'negative':
            df1 = df1.sort_values(by='{}-score'.format(x),ascending=True)
            df1 = df1.drop(['{}-score'.format(x)],axis=1)
            df1 = df1.iloc[:10]
            df1 = df1.sort_values(by='{}-counts'.format(x), ascending=False)
            return df1
        else:
            df1 = df1.sort_values(by='{}-score'.format(x), ascending=False)
            df1 = df1.drop(['{}-score'.format(x)], axis=1)
            df1 = df1.iloc[:10]
            df1 = df1.sort_values(by='{}-counts'.format(x), ascending=False)
            return df1

    list1 = []
    for x in x_data:
        df2 = main2(x)
        list1.append(df2)
    df3 = pd.concat(list1,axis=1)
    df3.to_csv('./重庆大学/情感高频特征词.csv',encoding='utf-8-sig',index=False)

def kmeans():
    f = open('./重庆大学/fenci.txt', 'w', encoding='utf-8')
    data = pd.read_csv('./重庆大学/重庆大学_情感分析.csv')
    data = data.dropna(how='any', axis=0)
    for line in data['分词']:
        text1 = line.split(' ')
        c = Counter()
        for x in text1:
                c[x] += 1
        # Top50
        output = ""
        # print('\n词频统计结果：')
        for (k, v) in c.most_common(30):
            # print("%s:%d"%(k,v))
            output += k + " "
        f.write(output + "\n")
    else:
        f.close()
    # 文档预料 空格连接
    corpus = []
    # 读取预料 一行预料为一个文档
    for line in open('./重庆大学/fenci.txt', 'r', encoding='utf-8').readlines():
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
    # weight1 = weight.astype('float16')

    n_clusters = 2
    # 打印特征向量文本内容
    print('Features length: ' + str(len(word)))
    # 第二步 聚类Kmeans
    print('Start Kmeans:')

    clf = KMeans(n_clusters=n_clusters)
    pre = clf.fit_predict(weight)
    print(pre)

    result = pd.concat((data, pd.DataFrame(pre)), axis=1)
    result.rename({0: '聚类结果'}, axis=1, inplace=True)
    result.to_csv('./重庆大学/聚类结果.csv', encoding="utf-8-sig")

    # 中心点
    print(clf.cluster_centers_)
    print(clf.inertia_)
    #
    # 第三步 图形输出 降维

    pca = PCA(n_components=n_clusters)  # 输出两维
    newData = pca.fit_transform(weight)  # 载入N维

    x = [n[0] for n in newData]
    y = [n[1] for n in newData]
    plt.figure(figsize=(12, 9), dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(x, y, c=pre, s=100)
    plt.title("词性聚类图")
    plt.savefig('./重庆大学/词性聚类图.png')
    plt.show()


def web_semantics():
    word = [] #记录关键词
    f = open("./重庆大学/fenci.txt", encoding='utf-8')
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

    f = open("./重庆大学/fenci.txt", encoding='utf-8')
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

    words = codecs.open("./重庆大学/word_node.txt", "w", "utf-8")
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
    with open('./重庆大学/word_node.txt','r',encoding='utf-8')as f:
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
    new_df = df.iloc[0:100]

    A = []
    B = []
    for w1,w2 in tqdm(zip(new_df['word1'],new_df['word2'])):
        if w1 != "" and w2 != "":
            A.append(w1)
            B.append(w2)
    elem_dic = tuple(zip(A,B))
    print(len(elem_dic))
    G = nx.Graph()
    G.add_edges_from(list(elem_dic))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.figure(figsize=(20, 14))
    pos=nx.spring_layout(G,iterations=10)
    nx.draw_networkx_nodes(G, pos, alpha=0.7,node_size=1600)
    nx.draw_networkx_edges(G,pos,width=0.5,alpha=0.8,edge_color='g')
    nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1,font_size='24')
    plt.title("网络语义")
    plt.savefig('./重庆大学/网络语义.png')
    plt.show()


def lda_tfidf():
    fr = open('./重庆大学/fenci.txt', 'r', encoding='utf-8')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ') if len(word) >= 2]
        train.append(line)

    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    # 构造主题数寻优函数
    def cos(vector1, vector2):  # 余弦相似度函数
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return (None)
        else:
            return (dot_product / ((normA * normB) ** 0.5))

        # 主题数寻优

    def lda_k(x_corpus, x_dict):
        # 初始化平均余弦相似度
        mean_similarity = []
        mean_similarity.append(1)

        # 循环生成主题并计算主题间相似度
        for i in np.arange(2, 11):
            lda = models.LdaModel(x_corpus, num_topics=i, id2word=x_dict)  # LDA模型训练
            for j in np.arange(i):
                term = lda.show_topics(num_words=50)

            # 提取各主题词
            top_word = []
            for k in np.arange(i):
                top_word.append([''.join(re.findall('"(.*)"', i)) \
                                 for i in term[k][1].split('+')])  # 列出所有词

            # 构造词频向量
            word = sum(top_word, [])  # 列出所有的词
            unique_word = set(word)  # 去除重复的词

            # 构造主题词列表，行表示主题号，列表示各主题词
            mat = []
            for j in np.arange(i):
                top_w = top_word[j]
                mat.append(tuple([top_w.count(k) for k in unique_word]))

            p = list(itertools.permutations(list(np.arange(i)), 2))
            l = len(p)
            top_similarity = [0]
            for w in np.arange(l):
                vector1 = mat[p[w][0]]
                vector2 = mat[p[w][1]]
                top_similarity.append(cos(vector1, vector2))

            # 计算平均余弦相似度
            mean_similarity.append(sum(top_similarity) / l)
        return (mean_similarity)

    # 计算主题平均余弦相似度
    word_k = lda_k(corpus, dictionary)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8), dpi=300)
    plt.plot(word_k)
    plt.title('LDA评论主题数寻优')
    plt.xlabel('主题数')
    plt.ylabel('平均余弦相似度')
    plt.savefig('./重庆大学/LDA评论主题数寻优.png')
    plt.show()

    topic_lda = input('请输入最优主题数:')
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_lda)
    data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(data, './重庆大学/lda.html')


if __name__ == '__main__':
    snownlp_fx()
    wordclound_fx()
    emotion_score()
    kmeans()
    web_semantics()
    lda_tfidf()