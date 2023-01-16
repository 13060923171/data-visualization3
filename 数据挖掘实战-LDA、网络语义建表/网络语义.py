import codecs
import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
import csv


def main1():
    word = []  # 记录关键词
    f = open("./data1/fenci.txt", encoding='utf-8-sig')
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
    print(len(word))  # 关键词总数

    # 类型<type 'numpy.ndarray'>
    word_vector = coo_matrix((len(word), len(word)), dtype=np.int8).toarray()
    print(word_vector.shape)

    f = open("./data1/fenci.txt", encoding='utf-8')
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
                # print n1, n2, w1, w2
                j = j + 1
            i = i + 1
        # 读取新内容
        line = f.readline()
    f.close()

    # --------------------------第三步  TXT文件写入--------------------------
    res = open("./data1/weight.txt", "a+", encoding='utf-8')
    i = 0
    while i < len(word):
        w1 = word[i]
        j = 0
        while j < len(word):
            w2 = word[j]
            # 判断两个词是否共现 共现&词频不为0的写入文件
            if word_vector[i][j] > 0:
                # print w1 +" " + w2 + " "+ str(int(word_vector[i][j]))
                res.write(w1 + " " + w2 + " " + str(int(word_vector[i][j])) + "\n")
            j = j + 1
        i = i + 1
    res.close()

    # --------------------------第四步  CSV文件写入--------------------------
    c = open("./data1/weight.csv", "w", encoding='gbk', newline='')  # 解决空行
    # c.write(codecs.BOM_UTF8)                                 #防止乱码
    writer = csv.writer(c)  # 写入对象
    writer.writerow(['Word1', 'Word2', 'Weight'])

    i = 0
    while i < len(word):
        w1 = word[i]
        j = 0
        while j < len(word):
            w2 = word[j]
            # 判断两个词是否共现 共现词频不为0的写入文件
            if word_vector[i][j] > 0:
                # 写入文件
                templist = []
                templist.append(w1)
                templist.append(w2)
                templist.append(str(int(word_vector[i][j])))
                # print templist
                writer.writerow(templist)
            j = j + 1
        i = i + 1
    c.close()


def main2():
    word = []  # 记录关键词
    f = open("./data2/fenci.txt", encoding='utf-8-sig')
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
    print(len(word))  # 关键词总数

    # 类型<type 'numpy.ndarray'>
    word_vector = coo_matrix((len(word), len(word)), dtype=np.int8).toarray()
    print(word_vector.shape)

    f = open("./data2/fenci.txt", encoding='utf-8')
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
                # print n1, n2, w1, w2
                j = j + 1
            i = i + 1
        # 读取新内容
        line = f.readline()
    f.close()

    # --------------------------第三步  TXT文件写入--------------------------
    res = open("./data2/weight.txt", "a+", encoding='utf-8')
    i = 0
    while i < len(word):
        w1 = word[i]
        j = 0
        while j < len(word):
            w2 = word[j]
            # 判断两个词是否共现 共现&词频不为0的写入文件
            if word_vector[i][j] > 0:
                # print w1 +" " + w2 + " "+ str(int(word_vector[i][j]))
                res.write(w1 + " " + w2 + " " + str(int(word_vector[i][j])) + "\n")
            j = j + 1
        i = i + 1
    res.close()

    # --------------------------第四步  CSV文件写入--------------------------
    c = open("./data2/weight.csv", "w", encoding='gbk', newline='')  # 解决空行
    # c.write(codecs.BOM_UTF8)                                 #防止乱码
    writer = csv.writer(c)  # 写入对象
    writer.writerow(['Word1', 'Word2', 'Weight'])

    i = 0
    while i < len(word):
        w1 = word[i]
        j = 0
        while j < len(word):
            w2 = word[j]
            # 判断两个词是否共现 共现词频不为0的写入文件
            if word_vector[i][j] > 0:
                # 写入文件
                templist = []
                templist.append(w1)
                templist.append(w2)
                templist.append(str(int(word_vector[i][j])))
                # print templist
                writer.writerow(templist)
            j = j + 1
        i = i + 1
    c.close()

def main3():
    data = pd.read_csv('./data1/weight.csv',encoding='gbk')
    df = pd.DataFrame()
    df['id'] = data['Word1']
    df['label'] = data['Word1']
    df = df.drop_duplicates(subset='id',keep='first')
    df = df.drop_duplicates(subset='label', keep='first')
    df = df.replace(to_replace="", value=np.NaN)
    df = df.dropna(how='any',axis=0)
    df.to_csv('./data1/entity.csv', encoding='gbk', index=False)

    df1 = pd.DataFrame()
    df1['Source'] = data['Word1']
    df1['Target'] = data['Word2']
    df1['Type'] = 'Undirected'
    df1['Weight'] = data['Weight']
    df1['Weight'] = df1['Weight'].astype(int)
    df1 = df1.replace(to_replace="",value=np.NaN)
    new_df = df1.dropna(how='any',axis=0)
    new_df1 = new_df[new_df['Weight'] >= 10]
    new_df1.to_csv('./data1/data1_weight.csv', encoding='gbk', index=False)


def main4():
    data = pd.read_csv('./data2/weight.csv',encoding='gbk')
    df = pd.DataFrame()
    df['id'] = data['Word1']
    df['label'] = data['Word1']
    df = df.drop_duplicates(subset='id',keep='first')
    df = df.drop_duplicates(subset='label', keep='first')
    df = df.replace(to_replace="", value=np.NaN)
    df = df.dropna(how='any',axis=0)
    df.to_csv('./data2/entity.csv', encoding='gbk', index=False)

    df1 = pd.DataFrame()
    df1['Source'] = data['Word1']
    df1['Target'] = data['Word2']
    df1['Type'] = 'Undirected'
    df1['Weight'] = data['Weight']
    df1['Weight'] = df1['Weight'].astype(int)
    df1 = df1.replace(to_replace="",value=np.NaN)
    new_df = df1.dropna(how='any',axis=0)
    new_df1 = new_df[new_df['Weight'] >= 10]
    new_df1.to_csv('./data2/data2_weight.csv', encoding='gbk', index=False)


if __name__ == '__main__':
    main1()
    main2()
    main3()
    main4()