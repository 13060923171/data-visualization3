import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import os
from tqdm import tqdm
from snownlp import SnowNLP

#使用自定义词典
jieba.load_userdict("custom_dict.txt")

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
            # if flag in ['Ag','a','ad','an','Ng','n','v','nr','ns','nt','nz']:
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
    converted_score = 2 * sentiment - 1
    if converted_score < 0:
        return converted_score
    else:
        return converted_score


# 定义一个函数，返回两个结果
def analyze_sentiment2(x):
    text = str(x)
    s = SnowNLP(text)
    sentiment = s.sentiments
    converted_score = 2 * sentiment - 1
    if converted_score < 0:
        return "负面"
    else:
        return "正面"


for i in range(1,13):
    df = pd.read_excel('2024年原始数据.xlsx',sheet_name=f'{i}月')
    # 初始化情感分析任务
    print('原数据总数:',len(df))
    df['评论内容'] = df['评论内容'].apply(preprocess_word)
    df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
    df = df.dropna(subset=['评论内容'], axis=0)
    df['fenci'] = df['评论内容'].apply(get_cut_words)
    df = df.dropna(subset=['fenci'], axis=0)
    print('清洗过后数据总数:',len(df))
    df['情感得分'] = df['fenci'].apply(analyze_sentiment1)
    df['情感类别'] = df['fenci'].apply(analyze_sentiment2)
    df.to_csv(f'{i}月.csv',index=False,encoding='utf-8-sig')