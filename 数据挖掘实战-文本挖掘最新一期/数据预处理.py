import pandas as pd
import os

list_df = []
filePath = './评论/国内/国内b站视频评论'
for i in os.listdir(filePath):
    data = pd.read_excel('./评论/国内/国内b站视频评论/{}'.format(i))
    list_df.append(data)

filePath1 = './评论/国内/国内快手评论'
for i in os.listdir(filePath1):
    data = pd.read_excel('./评论/国内/国内快手评论/{}'.format(i))
    list_df.append(data)

filePath2 = './评论/国内/国内抖音评论'
for i in os.listdir(filePath2):
    data = pd.read_excel('./评论/国内/国内抖音评论/{}'.format(i))
    list_df.append(data)

filePath3 = './评论/国外/Twitter评论'
for i in os.listdir(filePath3):
    data = pd.read_excel('./评论/国外/Twitter评论/{}'.format(i))
    list_df.append(data)

filePath4 = './评论/国外/YouTube评论'
for i in os.listdir(filePath4):
    data = pd.read_excel('./评论/国外/YouTube评论/{}'.format(i))
    list_df.append(data)

df = pd.concat(list_df,axis=0)
df = df['评论主体和次体'].drop_duplicates(keep='first')
df = df.dropna(how='any')
df1 = pd.DataFrame()
df1['内容'] = list(df.values)
df1.to_csv('数据集.csv',encoding='utf-8-sig')