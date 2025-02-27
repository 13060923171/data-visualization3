import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import os
from tqdm import tqdm
from paddlenlp import Taskflow

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
    # 进行情感分析
    results = sentiment_analysis(text)
    for result in results:
        label = result['label']
        score = result['score']
        return label,score


df = pd.read_excel('大众点评评论.xlsx')
# 初始化情感分析任务
sentiment_analysis = Taskflow("sentiment_analysis")
print('原数据总数:',len(df))
df['评论文本'] = df['评论文本'].apply(preprocess_word)
df['评论文本'] = df['评论文本'].apply(emjio_tihuan)
df = df.dropna(subset=['评论文本'], axis=0)
df['fenci'] = df['评论文本'].apply(get_cut_words)
df = df.dropna(subset=['fenci'], axis=0)
print('清洗过后数据总数:',len(df))
list_label = []
list_score = []
for d in df['fenci']:
    label,score = emotion_analysis(d)
    list_label.append(label)
    list_score.append(score)
df['label'] = list_label
df['score'] = list_score
df.to_csv('new_data.csv',index=False,encoding='utf-8-sig')
