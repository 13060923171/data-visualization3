import codecs
import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
import csv
import nltk
import re
from collections import Counter
from langdetect import detect
from tqdm import tqdm


def main1(df,name,emotion):
    new_df = df[df['情感分类'] == emotion]
    # 导入停用词列表
    keyword_words = []
    with open("./需求一/{}_{}_keyword.txt".format(name, emotion), 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            keyword_words.append(line.strip())

    def fenci():
        f = open('./需求一/{}_{}_fenci.txt'.format(name,emotion), 'w', encoding='utf-8-sig')
        for line in new_df['分词']:
            line = str(line)
            line = line.strip('\n')
            # 计算关键词
            all_words = line.split()
            c = Counter()
            for x in all_words:
                if x in keyword_words:
                    c[x] += 1
            output = ""
            for (k, v) in c.most_common():
                output += k + " "

            f.write(output + "\n")
        else:
            f.close()

    fenci()

    word = [] #记录关键词
    f = open("./需求一/{}_{}_fenci.txt".format(name,emotion), encoding='utf-8-sig')
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

    f = open("./需求一/{}_{}_fenci.txt".format(name,emotion), encoding='utf-8-sig')
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
    res = open("./需求一/{}_{}_weight.txt".format(name,emotion), "a+", encoding='utf-8-sig')
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
    c = open("./需求一/{}_{}_weight.csv".format(name,emotion), "w", encoding='gbk', newline='')  # 解决空行
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


def main3(name,emotion):
    data = pd.read_csv('./需求一/{}_{}_weight.csv'.format(name,emotion),encoding='gbk')
    df = pd.DataFrame()
    df['id'] = data['Word1']
    df['label'] = data['Word1']
    df = df.drop_duplicates(subset='id',keep='first')
    df = df.drop_duplicates(subset='label', keep='first')
    df = df.replace(to_replace="", value=np.NaN)
    df = df.dropna(how='any',axis=0)
    # df.to_csv('./需求一/{}_{}_entity.csv'.format(name,emotion), encoding='gbk', index=False)

    df1 = pd.DataFrame()
    df1['Source'] = data['Word1']
    df1['Target'] = data['Word2']
    df1['Type'] = 'Undirected'
    df1['Weight'] = data['Weight']
    df1['Weight'] = df1['Weight'].astype(int)
    df1 = df1.replace(to_replace="",value=np.NaN)
    new_df = df1.dropna(how='any',axis=0)
    # new_df1 = new_df[new_df['Weight'] >= 10]
    new_df.to_csv('./需求一/{}_{}_entity.csv'.format(name,emotion), encoding='gbk', index=False)


if __name__ == '__main__':
    list_time = ['前半年','后半年']
    list_emotion = ['正面情感', '负面情感']
    for t in list_time:
        for e in tqdm(list_emotion):
            df = pd.read_csv("{}数据.csv".format(t))
            main1(df,t,e)
            main3(t,e)
