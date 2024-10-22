import pandas as pd
import numpy as np

data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')

df = pd.concat([data1,data2],axis=0)
df = df.drop_duplicates(subset=['推文内容'])

def pandaun(s):
    s1 = float(s)
    if s1 >= 0.7:
        return s1
    else:
        return np.NAN



def time_process(x):
    x1 = str(x).split(" ")
    return x1[0]


df['时间'] = df['发布时间'].apply(time_process)
df['发帖数量'] = 1
df['准确率'] = df['score'].apply(pandaun)
df1 = df.dropna(subset=['准确率'],axis=0)
def df_process(data):
    data1 = data['label'].value_counts()
    d = {}
    for i,j in zip(data1.index,data1.values):
        d[i] = j
    return d

df11 = df1.groupby(by=['时间']).apply(df_process)
df3 = df.groupby(by=['时间']).agg({'发帖数量':'sum'})
df11 = df11.reset_index()
df11.columns = ['时间','类别']
df11 = df11.sort_index()

df4 = df3.reset_index()
df4.columns = ['时间','发帖数量']
df4 = df4.sort_index()

df5 = pd.merge(df11,df4,on='时间')

df2 = df5


list_0 = []
list_1 = []
list_2 = []
for d in df2['类别']:
    try:
        LABEL_0 = d['LABEL_0']
    except:
        LABEL_0 = 0
    try:
        LABEL_1 = d['LABEL_1']
    except:
        LABEL_1 = 0
    try:
        LABEL_2 = d['LABEL_2']
    except:
        LABEL_2 = 0
    list_0.append(LABEL_0)
    list_1.append(LABEL_1)
    list_2.append(LABEL_2)


df2['消极'] = list_0
df2['中立'] = list_1
df2['积极'] = list_2
df2 = df2.drop(['类别'],axis=1)
data = pd.read_excel('股票.xlsx')
df2['时间'] = pd.to_datetime(df2['时间'])
data['时间'] = pd.to_datetime(data['日付'])
data1 = pd.merge(df2,data,on='时间',how='right')
data1 = data1.dropna(how='any',axis=0)

data1.to_excel('情感分类数据.xlsx',index=False)
