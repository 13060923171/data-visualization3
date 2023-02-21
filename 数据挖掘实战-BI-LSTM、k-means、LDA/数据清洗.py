import pandas as pd
import numpy as np



#这里是文本清洗的工作，总的就是把一些空格，特殊字符等等全部给清洗一遍，就没啥好说的
data1 = pd.read_csv('./data/股票.csv')
data1['标签'] = '股票'
data2 = pd.read_csv('./data/#ChatGPT#.csv')
data2['标签'] = 'chatgpt'
data3 = pd.read_csv('./data/科比.csv')
data3['标签'] = '科比'
data4 = pd.read_csv('./data/詹姆斯历史得分王.csv')
data4['标签'] = '詹姆斯历史得分王'
df = pd.concat([data1,data2,data3,data4],axis=0)


def main1(x):
    x1 = str(x)
    x1 = x1.replace("'","").replace("[","").replace("]","").replace(" ","").replace("\n","")
    x1 = str(x1)
    x2 = x1.split('\\n')
    x2 = x2[0].replace('今天', '2022年02月10日')
    if '分钟' in x2:
        return '2022年02月10日'
    else:
        return x2


def main2(x):
    x1 = str(x)
    x1 = x1.replace("'","").replace("[","").replace("]","").replace(" ","").replace("\n","")
    x1 = str(x1)
    x2 = x1.split('\\n')
    return x2[0]


def main4(x):
    x1 = str(x)
    x1 = x1.replace("'","").replace("[","").replace("]","").replace(" ","").replace("\n","").replace("\u200b","")
    x1 = str(x1)
    x2 = x1.split('\\n')
    return x2[0]


def main5(x):
    x1 = str(x)
    x1 = x1.replace("'","").replace("[","").replace("]","").replace(" ","").replace("\n","").replace("\u200b","")
    if x1 == '赞':
        return 0
    elif len(x1) == 0:
        return 0
    else:
        return x1


def main6(x):
    x1 = str(x)
    x1 = x1.replace("'","").replace("[","").replace("]","").replace(" ","").replace("\n","").replace("\u200b","").replace(",","")
    if x1 == '转发':
        return 0
    elif len(x1) == 0:
        return 0
    else:
        return x1


def main7(x):
    x1 = str(x)
    x1 = x1.replace("'","").replace("[","").replace("]","").replace(" ","").replace("\n","").replace("\u200b","").replace(",","")
    if x1 == '评论':
        return 0
    elif len(x1) == 0:
        return 0
    else:
        return x1


df['时间'] = df['时间'].apply(main1)
df['博主'] = df['博主'].apply(main2)
df['内容'] = df['内容'].apply(main4)
df['点赞'] = df['点赞'].apply(main5)
df['转发'] = df['转发'].apply(main6)
df['评论'] = df['评论'].apply(main7)

df = df.drop_duplicates(subset=['内容'],keep='first')
# df = df.dropna(how='any',axis=0)
df.to_csv('./data/data.csv',encoding='utf-8-sig',index=None)