import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import os
from tqdm import tqdm


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


# 判断是否为中文，或处于允许列表中的词汇（如 "AI"）
def is_all_chinese_or_allowed(strs, allowed_words):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5' and strs not in allowed_words:
            return False
    return True


def get_cut_words(content_series, allowed_words=['AI', 'ai']):
    try:
        # 对文本进行分词和词性标注
        words = pseg.cut(content_series)
        # 保存名词和形容词的列表
        nouns_and_adjs = []
        # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
        for word, flag in words:
            if flag in ['Ag', 'a', 'ad', 'an', 'Ng', 'n', 'v']:
                if word not in stop_words and len(word) >= 2 and is_all_chinese_or_allowed(word, allowed_words):
                    # 如果是名词或形容词，就将其保存到列表中
                    nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except Exception as e:
        print(f"Error: {e}")
        return np.NAN


df1 = pd.read_excel('微博.xlsx',sheet_name='消费模式')
df2 = pd.read_excel('微博.xlsx',sheet_name='人工智能')
df3 = pd.read_excel('微博.xlsx',sheet_name='新闻生产')
df = pd.concat([df1,df2,df3],axis=0)
print('原数据总数:',len(df))
df = df.drop_duplicates(subset=['评论内容'])
df['评论内容'] = df['评论内容'].apply(preprocess_word)
df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
df = df.dropna(subset=['评论内容'], axis=0)
df['分词'] = df['评论内容'].apply(get_cut_words)
df = df.dropna(subset=['分词'], axis=0)
print('清洗过后数据总数:',len(df))
df.to_excel('new_data.xlsx',index=False)