import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import os
from tqdm import tqdm
from snownlp import SnowNLP

# 导入停用词列表
stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


#去掉标点符号，以及机械压缩
def preprocess_word(word):
    word1 = str(word)
    word1 = re.sub(r'#\w+#', '', word1)
    word1 = re.sub(r'【.*?】', '', word1)
    word1 = re.sub(r'@[\w]+', '', word1)
    word1 = re.sub(r'[a-zA-Z]', '', word1)
    word1 = re.sub(r'\.\d+', '', word1)
    return word1


def emjio_tihuan(x):
    x1 = str(x)
    x2 = re.sub('(\[.*?\])', "", x1)
    x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
    x4 = re.sub(r'\n', '', x3)
    return x4

# 定义机械压缩函数
def yasuo(st):
    for i in range(1, int(len(st) / 2) + 1):
        for j in range(len(st)):
            if st[j:j + i] == st[j + i:j + 2 * i]:
                k = j + i
                while st[k:k + i] == st[k + i:k + 2 * i] and k < len(st):
                    k = k + i
                st = st[:j] + st[k:]
    return st


# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def get_cut_words(content_series):
    try:
        # 对文本进行分词和词性标注
        words = pseg.cut(content_series)
        # 保存名词和形容词的列表
        nouns_and_adjs = []
        # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
        for word, flag in words:
            #判断是否为名词或者形容词或者动词
            # if flag in ['Ag','a','ad','an','Ng','n','ns','nz','nt','v']:
            if flag in ['Ag','a','ad','an']:
                if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                    # 如果是名词或形容词，就将其保存到列表中
                    nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN


# 定义一个函数，返回两个结果
def analyze_sentiment1(x):
    text = str(x)
    s = SnowNLP(text)
    sentiment = s.sentiments
    if sentiment <= 0.2:
        return '强烈消极'
    elif 0.2 < sentiment <= 0.35:
        return '消极'
    elif 0.35 < sentiment <= 0.5:
        return '比较消极'
    elif 0.5 < sentiment <= 0.6:
        return '中立'
    elif 0.6 < sentiment <= 0.75:
        return '比较积极'
    elif 0.75 < sentiment <= 0.9:
        return '积极'
    else:
        return '强烈积极'

df1 = pd.read_excel('马蜂窝.xlsx')
content1 = df1['正文']
df3 = pd.read_excel('穷游游记.xlsx')
content3 = df3['正文']
df4 = pd.read_excel('去哪儿.xlsx')
content4 = df4['游记正文']
df5 = pd.read_excel('小红书.xlsx')
content5 = df5['正文']
df6 = pd.read_excel('携程.xlsx')
content6 = df6['正文']

df2 = pd.read_excel('穷游评论.xlsx')
content2 = df2['评论内容']
df7 = pd.read_excel('携程景区评论.xlsx')
content7 = df7['评价内容']
df8 = pd.read_excel('携程评论数据2.xlsx')
content8 = df8['评价内容']

# data = pd.concat([content1,content3,content4,content5,content6],axis=0)
data = pd.concat([content2,content7,content8],axis=0)
df = pd.DataFrame()
df['内容'] = data.values

df = df.drop_duplicates(subset=['内容'])
print('原数据总数:',len(df))
df['内容'] = df['内容'].apply(preprocess_word)
df['内容'] = df['内容'].apply(emjio_tihuan)
df = df.dropna(subset=['内容'], axis=0)
# df['内容'] = df['内容'].apply(yasuo)
df['fenci'] = df['内容'].apply(get_cut_words)
df = df.dropna(subset=['fenci'], axis=0)
df['情感类别'] = df['fenci'].apply(analyze_sentiment1)
print('清洗过后数据总数:',len(df))
df.to_csv('评论数据2.csv',index=False,encoding='utf-8-sig')