import codecs
import networkx as nx
import matplotlib.pyplot as plt
import csv
import pandas as pd
import nltk
from collections import Counter
from scipy.sparse import coo_matrix
from tqdm import tqdm
import numpy as np
import re
import jieba
from tqdm import tqdm
import concurrent.futures
import os


def main(name):
    df = pd.read_csv('./数据集/{}.csv'.format(name))

    # 导入停用词列表
    stop_words = []
    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    # 判断是否为中文
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    def fenci():
        f = open('./data_txt/{}_fenci.txt'.format(name), 'w', encoding='utf-8-sig')
        for line in df['分词']:
            line = str(line)
            line = line.strip('\n')
            # 计算关键词
            all_words = line.split()
            c = Counter()
            for x in all_words:
                c[x] += 1
            output = ""
            for (k, v) in c.most_common(30):
                output += k + " "

            f.write(output + "\n")
        else:
            f.close()

    fenci()

    word = [] #记录关键词
    f = open("./data_txt/{}_fenci.txt".format(name), encoding='utf-8')
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

    f = open("./data_txt/{}_fenci.txt".format(name), encoding='utf-8')
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

    words = codecs.open("./data_txt/{}_word_node.txt".format(name), "w", "utf-8")
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
    with open('./data_txt/{}_word_node.txt'.format(name),'r',encoding='utf-8')as f:
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
    new_df = df.iloc[0:80]

    A = []
    B = []
    for w1,w2 in tqdm(zip(new_df['word1'],new_df['word2'])):
        if w1 != "" and w2 != "":
            A.append(w1)
            B.append(w2)
    elem_dic = tuple(zip(A,B))
    print(len(elem_dic))
    #创建一个空的无向图。即创建了一个称为G的图对象，用于保存文本数据的节点和边信息。
    G = nx.Graph()
    #向图G中添加节点和边。这里的list(elem_dic)表示将elem_dic字典中的元素列表作为图的边。其中elem_dic字典中存储着文本数据的节点和边信息。
    G.add_edges_from(list(elem_dic))
    #设置图像中使用中文字体，以避免出现显示中文乱码的情况。这里将字体设置为SimHei，使用sans-serif字体族。
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'sans-serif'
    #设置图像的大小，其中figsize参数设置图像的宽度和高度。
    plt.figure(figsize=(16, 9),dpi=500)
    #确定节点布局。这里使用了一种称为spring layout的布局算法，相当于在二维空间中对节点进行排列。iterations参数指定了进行节点排列的迭代次数。
    pos=nx.spring_layout(G,iterations=10)
    #绘制节点。其中alpha参数设置节点的透明度，node_size参数设置节点的大小。
    nx.draw_networkx_nodes(G, pos, alpha=0.7,node_size=800)
    #绘制边。其中width参数设置边的宽度，alpha参数设置边的透明度，edge_color参数设置边的颜色。
    nx.draw_networkx_edges(G,pos,width=0.5,alpha=0.8,edge_color='g')
    #添加标签。其中font_family参数指定图像中使用sans-serif字体族，alpha参数设置节点标签的透明度，font_size参数设置归纳节点标签的字体大小。
    nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1,font_size='10')
    plt.title("co-occurrence semantics")
    plt.savefig('./img/{}_共现语义.png'.format(name))
    # plt.show()


if __name__ == '__main__':
    list_name = ['阿凡达2：水之道', '速度与激情10', '银河护卫队3', '奥本海默', '变形金刚7：超能勇士崛起', '蜘蛛侠：纵横宇宙', '蚁人3', '芭比', '超级马里奥', '黑豹2',
                 '闪电侠', '龙与地下城', '雷霆沙赞', '疯狂元素城', '小美人鱼',
                 '阿凡达1', '速度与激情9', '银河护卫队2', '信条', '大黄蜂', '蜘蛛侠：平行宇宙', '蚁人2', '黑豹', '神奇女侠1984', '疯狂动物城', '美女与野兽', '长津湖',
                 '战狼2', '你好李焕英', '哪吒之魔童降世', '流浪地球', '满江红',
                 '唐人街探案3', '复仇者联盟4终局之战', '长津湖之水门桥', '流浪地球2', '红海行动']

    def list_files_in_directory(directory):
        return os.listdir(directory)

    # 使用
    folder_path = "img"
    list_name1 = list_files_in_directory(folder_path)
    list_name2 = []
    for l in list_name1:
        l = str(l).replace('_共现语义.png','')
        list_name2.append(l)

    list_name3 = list(set(list_name).difference(list_name2))

    for l in tqdm(list_name3):
        main(l)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=4)as e:
    #     # 通过 map function 来并发启动加载url的任务，并收集结果到future_to_url
    #     future_to_url = [e.submit(main, l) for l in list_name[14:]]
    #     for future in tqdm(concurrent.futures.as_completed(future_to_url),total=len(future_to_url)):
    #         print(futuer.result())