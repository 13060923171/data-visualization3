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
import threading as th


def main():
    with open('class-fenci.txt','r',encoding='utf-8-sig')as f:
        content = f.readlines()
    cut_word_list = list(map(lambda x: ''.join(x), content))
    content_str = ' '.join(cut_word_list).split()
    d = {}
    for c in content_str:
        d[c] = d.get(c,0)+1
    ls = list(d.items())
    ls.sort(key=lambda x:x[1],reverse=True)
    x_data = []
    y_data = []
    for key,values in ls:
        x_data.append(key)
        y_data.append(values)

    x_data1 = x_data[:50]
    matrix = np.zeros((len(x_data1)+1)*(len(x_data1)+1)).reshape(len(x_data1)+1, len(x_data1)+1).astype(str)
    matrix[0][0] = np.NaN
    matrix[1:, 0] = matrix[0, 1:] = x_data1
    cont_list = [cont.split() for cont in cut_word_list]
    for i, w1 in enumerate(x_data1):
        for j, w2 in enumerate(x_data1):
            count = 0
            for cont in cont_list:
                if w1 in cont and w2 in cont:
                    if abs(cont.index(w1)-cont.index(w2)) == 0 or abs(cont.index(w1)-cont.index(w2)) == 1:
                        count += 1
            matrix[i+1][j+1] = count

    kwdata = pd.DataFrame(data=matrix)
    kwdata.to_csv('关键词共现矩阵.csv', index=False, header=False, encoding='utf-8-sig')

    df = pd.read_csv('关键词共现矩阵.csv')
    df.index = df.iloc[:, 0].tolist()
    df_ = df.iloc[:50, 1:51]
    df_.astype(int)

    plt.figure(figsize=(10, 10))
    graph1 = nx.from_pandas_adjacency(df_)

    nx.draw(graph1, with_labels=True, node_color='blue',alpha=0.8, font_size=20, edge_color='#E3DC9D')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("共现网络图")
    plt.savefig('共现网络图.png')


if __name__ == '__main__':
    main()