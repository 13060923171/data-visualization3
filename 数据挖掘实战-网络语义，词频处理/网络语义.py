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
import itertools
from tqdm import tqdm
import os
import codecs
import networkx as nx
from scipy.sparse import coo_matrix
import jieba
import jieba.analyse

def web_semantics():
    df = pd.read_csv('new_data.csv')
    data = df.drop_duplicates(keep='first')

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

        # 读入停用词表

    stop_words = []

    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    data['fulltext'] = data['fulltext'].apply(emjio_tihuan)
    data = data.dropna(subset=['fulltext'], axis=0)

    #数据分词，把数据保存为txt文件
    f = open('fenci.txt', 'w', encoding='utf-8')
    for line in data['fulltext']:
        res = jieba.lcut(line, cut_all=False)
        c = Counter()
        for x in res:
            if x not in stop_words and len(x) >= 2 and is_all_chinese(x) == True:
                c[x] += 1
        # Top30
        output = ""
        # print('\n词频统计结果：')
        for (k, v) in c.most_common(30):
            output += k + " "
        f.write(output + "\n")
    else:
        f.close()

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
    #创建共线矩阵
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
    #计算词的权重
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
    #把词权重按照从大到小的顺序排序
    df = df.sort_values(by=['weight'],ascending=False)
    df = df.dropna(how='any',axis=1)
    new_df = df.iloc[:300]
    #删除重复词
    A = []
    B = []
    for w1,w2 in tqdm(zip(new_df['word1'],new_df['word2'])):
        if w1 != "" and w2 != "":
            A.append(w1)
            B.append(w2)
    elem_dic = tuple(zip(A,B))
    print(len(elem_dic))
    #画图，构建网络语义图
    G = nx.Graph()
    G.add_edges_from(list(elem_dic))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.figure(figsize=(20, 14))
    #设置格式
    pos=nx.spring_layout(G,iterations=10)
    #设置词的大小
    nx.draw_networkx_nodes(G, pos, alpha=0.7,node_size=1600)
    #设置线的大小
    nx.draw_networkx_edges(G,pos,width=0.5,alpha=0.8,edge_color='g')
    #设置文字的大小
    nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1,font_size='24')
    plt.title("网络语义")
    plt.savefig('网络语义.png')
    plt.show()


if __name__ == '__main__':
    web_semantics()