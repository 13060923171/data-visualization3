import numpy as np
import pandas as pd

data1 = pd.read_excel('data.xlsx')
data2 = pd.read_excel('data1.xlsx')
df1 = pd.concat([data1,data2],axis=0)
df2 = pd.read_excel('评论特征.xlsx',sheet_name='聚类一')
df2_word = [word for word in df2['word']]
df3 = pd.read_excel('评论特征.xlsx',sheet_name='聚类二')
df3_word = [word for word in df3['word']]


def demo1(x):
    x1 = str(x)
    x1_word = []
    for i in df2_word:
        if i in x1:
            x1_word.append(i)
    return len(x1_word)

def demo2(x):
    x1 = str(x)
    x1_word = []
    for i in df3_word:
        if i in x1:
            x1_word.append(i)
    return len(x1_word)



df1['聚类一'] = df1['帖子正文'].apply(demo1)
df1['聚类二'] = df1['帖子正文'].apply(demo2)
df1['一级特征'] = df1['聚类一'] + df1['聚类二']
df1['一级特征'] = df1['一级特征'].replace(0,np.NaN)
df1 = df1.dropna(subset=['一级特征'],axis=0)
df1.to_excel('特征词.xlsx')