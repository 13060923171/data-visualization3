import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
from snownlp import SnowNLP
import threading
import concurrent.futures
from tqdm import tqdm
import os


def demo1():
    df1 = pd.read_excel('./b站/评论下载.xlsx')
    df2 = df1.dropna(subset=['评论内容'], axis=0)
    return df2['评论内容']


def demo2():
    df1 = pd.read_excel('./b站/评论下载.xlsx')
    df2 = df1.dropna(subset=['评论回复'], axis=0)
    return df2['评论回复']

data1 = demo1()
data2 = demo2()

data = pd.concat([data1,data2],axis=0)
# 去除重复行
data = data.drop_duplicates()

df = pd.DataFrame()
df['评论内容'] = data.values


# 导入停用词列表
stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


#去掉标点符号，以及机械压缩
def preprocess_word(word):
    word1 = str(word)
    word1 = re.sub(r'转发微博', '', word1)
    word1 = re.sub(r'#\w+#', '', word1)
    word1 = re.sub(r'【.*?】', '', word1)
    word1 = re.sub(r'回复 @[\w]+', '', word1)
    word1 = re.sub(r'[a-zA-Z]', '', word1)
    word1 = re.sub(r'\.\d+', '', word1)
    return word1


def emjio_tihuan(x):
    x1 = str(x)
    x2 = re.sub('(\[.*?\])', "", x1)
    x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
    x4 = re.sub(r'\n', '', x3)
    return x4

# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def get_cut_words(content_series):
    # 对文本进行分词和词性标注
    try:
        words = pseg.cut(content_series)
        # 保存名词和形容词的列表
        nouns_and_adjs = []
        # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
        for word, flag in words:
            if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                # 如果是名词或形容词，就将其保存到列表中
                nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN


def snownlp_content(text):
    # 初始化SnowNLP对象
    try:
        s1 = SnowNLP(text)
        number = s1.sentiments
        number = float(number)
        if 0.45 <= number <= 0.55:
            return '中立'
        elif number > 0.55:
            return '正面'
        else:
            return '负面'
    except:
        return '中立'


df = df.dropna(how='any',axis=0)
df['评论内容'] = df['评论内容'].apply(preprocess_word)
df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
df['情感类型'] = df['评论内容'].apply(snownlp_content)
df['分词'] = df['评论内容'].apply(get_cut_words)
df = df.dropna(subset=['分词'], axis=0)
df.to_csv('./B站数据/B站数据.csv', encoding='utf-8-sig', index=False)






