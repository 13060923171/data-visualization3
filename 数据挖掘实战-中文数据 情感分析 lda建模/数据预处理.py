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
    word1 = re.sub(r'回复@[\w]+', '', word1)
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


# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def get_cut_words1(content_series):
    try:
        # 对文本进行分词和词性标注
        words = pseg.cut(content_series)
        # 保存名词和形容词的列表
        nouns_and_adjs = []
        # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
        for word, flag in words:
            if flag in ['Ag','a','ad','an','Ng','n','v']:
                if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                    # 如果是名词或形容词，就将其保存到列表中
                    nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN

def get_cut_words2(content_series):
    try:
        # 对文本进行分词和词性标注
        words = pseg.cut(content_series)
        # 保存名词和形容词的列表
        nouns_and_adjs = []
        # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
        for word, flag in words:
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


def emotion_analysis(text):
    s = SnowNLP(text)
    sentiment = s.sentiments
    if sentiment < 0.4:
        return "负面"
    elif 0.4 <= sentiment <= 0.6:
        return "中立"
    else:
        return "正面"


df1 = pd.read_excel('评论表1.xlsx')
df2 = pd.read_excel('评论表2.xlsx')
df = pd.concat([df1,df2],axis=0)
df = df.drop_duplicates(subset=['评论内容'])

print('原数据总数:',len(df))
df['评论内容'] = df['评论内容'].apply(preprocess_word)
df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
df = df.dropna(subset=['评论内容'], axis=0)
df['fenci'] = df['评论内容'].apply(get_cut_words1)
df['情感词'] = df['评论内容'].apply(get_cut_words2)
df = df.dropna(subset=['fenci'], axis=0)
print('清洗过后数据总数:',len(df))
df['label'] = df['fenci'].apply(emotion_analysis)
df.to_excel('评论表.xlsx',index=False)
