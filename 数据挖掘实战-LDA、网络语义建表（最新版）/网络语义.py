import codecs
import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np


def main1():
    word = [] #记录关键词
    f = open('./data1/fenci.txt', encoding='utf-8-sig')
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

    f = open('./data1/fenci.txt', encoding='utf-8-sig')
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

    words = codecs.open("./data1/word_node.txt", "w", "utf-8")
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

    with open("./data1/word_node.txt",'r',encoding='utf-8')as f:
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
    df = df.replace(to_replace="",value=np.NaN)
    new_df = df.dropna(how='any',axis=0)
    new_df1 = new_df[new_df['weight'] >= 10]
    new_df1.to_csv('./data1/node_word.csv', encoding='utf-8-sig', index=False)


def main2():
    word = [] #记录关键词
    f = open('./data2/fenci.txt', encoding='utf-8-sig')
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

    f = open('./data2/fenci.txt', encoding='utf-8-sig')
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

    words = codecs.open("./data2/word_node.txt", "w", "utf-8")
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

    with open("./data2/word_node.txt",'r',encoding='utf-8')as f:
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
    df = df.replace(to_replace="",value=np.NaN)
    new_df = df.dropna(how='any',axis=0)
    new_df1 = new_df[new_df['weight'] >= 10]
    new_df1.to_csv('./data2/node_word.csv', encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    main1()
    main2()
