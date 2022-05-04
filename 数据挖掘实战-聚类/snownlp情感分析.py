import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from snownlp import SnowNLP
import re

df = pd.read_excel('中国新闻网评论.xlsx')

content1 = df['评论内容'].drop_duplicates(keep='first')
content2 = df['评论内容.1'].drop_duplicates(keep='first')

content3 = pd.concat([content1,content2],axis=0)
content3 = content3.dropna(how='any')


def emotion_scroe(x):
    text = re.sub(r'(?:回复)?(?://)?@[\w\u2E80-\u9FFF]+:?|\[\w+\]', ',', x)
    score = SnowNLP(text)
    fenshu = score.sentiments - 0.5
    return fenshu

def wenchu(x):
    text = re.sub(r'(?:回复)?(?://)?@[\w\u2E80-\u9FFF]+:?|\[\w+\]', ',', x)
    return text

df1 = pd.DataFrame()
df1['content'] = content3.apply(wenchu)
df1['emotion_scroe'] = df1['content'].apply(emotion_scroe)
df1.to_csv('snownlp情感分析.csv',encoding='utf-8-sig')


plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(12,6),dpi=300)
plt.hist(df1['emotion_scroe'], bins=np.arange(-0.5, 0.5, 0.01), facecolor='#E74C3C')
plt.xlabel('情感数值')
plt.ylabel('数量')
plt.title('情感分析')
plt.savefig('Analysis of Sentiments.jpg')
plt.show()