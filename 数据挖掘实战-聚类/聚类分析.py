import pandas as pd
# 数据处理库
import re
import jieba
import time
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType
import codecs
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import jieba.analyse

df = pd.read_excel('中国新闻网评论.xlsx')

content1 = df['评论内容'].drop_duplicates(keep='first')
content2 = df['评论内容.1'].drop_duplicates(keep='first')

content3 = pd.concat([content1,content2],axis=0)
content3 = content3.dropna(how='any')


#删除标点符号
def clear_characters(text):
    return re.sub('\W', '', text)


#定义机械压缩函数
def yasuo(st):
    for i in range(1,int(len(st)/2)+1):
        for j in range(len(st)):
            if st[j:j+i] == st[j+i:j+2*i]:
                k = j + i
                while st[k:k+i] == st[k+i:k+2*i] and k<len(st):
                    k = k + i
                st = st[:j] + st[k:]
    return st

content3 = content3.astype(str)
# content3 = content3.apply(clear_characters)
content3 = content3.apply(yasuo)



# ------------------------------------中文分词------------------------------------
cut_words = ""
all_words = ""
f = open('fenci.txt', 'w', encoding='utf-8')

jieba.analyse.set_stop_words('stopwords_cn.txt')
for line in content3:
    line = line.strip('\n')
    # 停用词过滤
    line = re.sub(r'(?:回复)?(?://)?@[\w\u2E80-\u9FFF]+:?|\[\w+\]', ',', line)
    # line = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", line)
    seg_list = jieba.cut(line, cut_all=False)
    cut_words = (" ".join(seg_list))

    # 计算关键词
    all_words = cut_words.split()
    c = Counter()
    for x in all_words:
        if len(x) > 1 and x != '\r\n':
            c[x] += 1
    output = ""
    for (k, v) in c.most_common():
        # print("%s:%d"%(k,v))
        output += k + " "

    f.write(output + "\n")
else:
    f.close()




#
# #第一步 计算TFIDF
# # 文档预料 空格连接
# corpus = []
#
# # 读取预料 一行预料为一个文档
# for line in open('fenci.txt', 'r',encoding='utf-8').readlines():
#     corpus.append(line.strip())
# # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
# vectorizer = CountVectorizer()
#
# # 该类会统计每个词语的tf-idf权值
# transformer = TfidfTransformer()
#
# # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
# tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# # 获取词袋模型中的所有词语
# word = vectorizer.get_feature_names()
#
# # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
# weight = tfidf.toarray()
#
#
# data = {'word': word,
#         'tfidf': weight.sum(axis=0).tolist()}
# df2 = pd.DataFrame(data)
# df2['tfidf'] = df2['tfidf'].astype('float64')
# df2 = df2.sort_values(by=['tfidf'],ascending=False)
# df2.to_csv('tfidf.csv',encoding='utf-8-sig')
# c = (
#     WordCloud()
#         .add("",[(i, int(j)) for i, j in zip(df2['word'][:100], df2['tfidf'][:100])], word_size_range=[20, 100], shape=SymbolType.DIAMOND)
#         .set_global_opts(title_opts=opts.TitleOpts(title="WordCloud-shape-diamond"))
#         .render("TFIDF-词云.html")
# )
#
# result = codecs.open('文本向量化.txt', 'w', 'utf-8')
# # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
# for i in range(len(weight)):
#     # print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"
#     for j in range(len(word)):
#         # print weight[i][j],
#         result.write(str(weight[i][j]) + ' ')
#     result.write('\r\n\r\n')
# result.close()
# # 打印特征向量文本内容
# print('Features length: ' + str(len(word)))
#
# #第二步 聚类Kmeans
#
# print('Start Kmeans:')
# from sklearn.cluster import KMeans
#
# clf = KMeans(n_clusters=3)
# print(clf)
# pre = clf.fit_predict(weight)
# print(pre)
# with open('fenci.txt','r',encoding='utf-8')as f:
#     content = f.readlines()
#
# list1 = []
# for c in content:
#     c = c.strip('\n').strip(' ')
#     list1.append(c)
# print(len(list1))
# print(len(pre))
# df3 = pd.DataFrame()
# df3['word'] = list1
# result = pd.concat((df3, pd.DataFrame(pre)), axis=1)
# result.rename({0: '聚类结果'}, axis=1, inplace=True)
# result.to_csv('聚类分类.csv',encoding="utf-8-sig")
#
# # 中心点
# print(clf.cluster_centers_)
# print(clf.inertia_)
#
# #第三步 图形输出 降维
#
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=3)  # 输出两维
# newData = pca.fit_transform(weight)  # 载入N维
#
# x = [n[0] for n in newData]
# y = [n[1] for n in newData]
# plt.figure(figsize=(9,6),dpi = 300)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
# plt.rcParams['axes.unicode_minus'] = False
# plt.scatter(x, y, c=pre, s=100)
# plt.title("词性聚类图")
# plt.savefig('词性聚类图.jpg')
# plt.show()