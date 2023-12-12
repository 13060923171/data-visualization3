import codecs
import networkx as nx
from collections import Counter
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np


def demo(time_data):
    df1 = pd.read_csv('reply_data.csv')
    # df = df1[df1['时间'] == time_data]
    df = df1
    # d = {}
    # list_text = []
    # for t in df['分词']:
    #     t1 = str(t).split(" ")
    #     # 把数据分开
    #     for i in t1:
    #         # 添加到列表里面
    #         list_text.append(i)
    #         d[i] = d.get(i,0)+1
    #
    # ls = list(d.items())
    # ls.sort(key=lambda x:x[1],reverse=True)
    # x_data = []
    # y_data = []
    # for key,values in ls[:100]:
    #     x_data.append(key)
    #     y_data.append(values)
    #
    # data = pd.DataFrame()
    # data['word'] = x_data
    # data['counts'] = y_data

    x_data = []
    with open("custom_dict.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            x_data.append(line.strip())

    # 导入停用词列表

    def fenci():
        f = open('{}_fenci.txt'.format(time_data), 'w', encoding='utf-8-sig')
        for line in df['分词']:
            line1 = str(line).split(' ')
            # 计算关键词
            c = Counter()
            for x in line1:
                if x in x_data:
                    c[x] += 1
            output = ""
            for (k, v) in c.most_common():
                output += k + " "

            f.write(output + "\n")
        else:
            f.close()

    fenci()

    word = []  # 记录关键词
    f = open("{}_fenci.txt".format(time_data), encoding='utf-8')
    line = f.readline()
    while line:
        # print line
        line = line.replace("\n", "")  # 过滤换行
        line = line.strip('\n')
        for n in line.split(' '):
            # print n
            if n not in word:
                word.append(n)
        line = f.readline()
    f.close()

    word_vector = coo_matrix((len(word), len(word)), dtype=np.int8).toarray()

    f = open("{}_fenci.txt".format(time_data), encoding='utf-8')
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

    words = codecs.open("{}_word_node.txt".format(time_data), "w", "utf-8")
    i = 0
    while i < len(word):
        student1 = word[i]
        j = i + 1
        while j < len(word):
            student2 = word[j]
            if word_vector[i][j] > 0:
                words.write(student1 + " " + student2 + " "
                            + str(word_vector[i][j]) + "\r\n")
            j = j + 1
        i = i + 1
    words.close()

    """ 第四步:图形生成 """
    with open('{}_word_node.txt'.format(time_data), 'r', encoding='utf-8') as f:
        content = f.readlines()
    list_word1 = []
    list_word2 = []
    list_weight = []
    for i in content:
        c = i.strip('\n').split(" ")
        list_word1.append(c[0])
        list_word2.append(c[1])
        list_weight.append(c[2])

    data1 = pd.DataFrame()
    data1['word1'] = list_word1
    data1['word2'] = list_word2
    data1['weight'] = list_weight
    data1['weight'] = data1['weight'].astype(int)
    data1 = data1.sort_values(by=['weight'], ascending=False)
    data1 = data1.dropna(how='any', axis=1)
    new_data = data1.iloc[:150]

    A = []
    B = []
    for w1, w2 in tqdm(zip(new_data['word1'], new_data['word2'])):
        if w1 != "" and w2 != "":
            A.append(w1)
            B.append(w2)
    elem_dic = tuple(zip(A, B))
    # 创建一个空的无向图。即创建了一个称为G的图对象，用于保存文本数据的节点和边信息。
    G = nx.Graph()
    # 向图G中添加节点和边。这里的list(elem_dic)表示将elem_dic字典中的元素列表作为图的边。其中elem_dic字典中存储着文本数据的节点和边信息。
    G.add_edges_from(list(elem_dic))
    # 设置图像中使用中文字体，以避免出现显示中文乱码的情况。这里将字体设置为SimHei，使用sans-serif字体族。
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'sans-serif'
    # 设置图像的大小，其中figsize参数设置图像的宽度和高度。
    plt.figure(figsize=(16, 9), dpi=500)
    # 确定节点布局。这里使用了一种称为spring layout的布局算法，相当于在二维空间中对节点进行排列。iterations参数指定了进行节点排列的迭代次数。
    pos = nx.spring_layout(G, iterations=10)
    # 绘制节点。其中alpha参数设置节点的透明度，node_size参数设置节点的大小。
    nx.draw_networkx_nodes(G, pos, alpha=0.7, node_size=800)
    # 绘制边。其中width参数设置边的宽度，alpha参数设置边的透明度，edge_color参数设置边的颜色。
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8, edge_color='g')
    # 添加标签。其中font_family参数指定图像中使用sans-serif字体族，alpha参数设置节点标签的透明度，font_size参数设置归纳节点标签的字体大小。
    nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1, font_size='10')
    plt.title("{}_共现语义".format(time_data))
    plt.savefig('{}_共现语义.png'.format(time_data))
    plt.show()


if __name__ == '__main__':
    # list_data = [2020,2021,2022,2023]
    # for l in tqdm(list_data):
    #     demo(l)
    demo('评论')