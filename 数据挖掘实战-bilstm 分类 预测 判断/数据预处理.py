import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
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
    # word1 = re.sub(r'转发微博', '', word1)
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


def get_cut_words(content_series):
    try:
        # 对文本进行分词和词性标注
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

def process(x):
    #0 1 2 0是中立 1是正面 2是负面
    x1 = str(x)
    if x1 == '-1.0':
        return 2
    elif x1 == '0.0':
        return 0
    elif x1 == '1.0':
        return 1
    else:
        return x1
df = pd.read_csv('data.csv')
df['微博全文'] = df['微博全文'].apply(preprocess_word)
df['微博全文'] = df['微博全文'].apply(emjio_tihuan)
df.dropna(subset=['微博全文'], axis=0,inplace=True)
df['分词'] = df['微博全文'].apply(get_cut_words)
df['情感极性'] = df['情感极性'].apply(process)
new_df = df.dropna(subset=['分词'], axis=0)


def process1(x):
    if str(x) != 'nan':
        return '训练集'
    else:
        return '测试集'

new_df['训练集区分'] = new_df['情感极性'].apply(process1)

new_df1 = new_df[new_df['训练集区分'] == '训练集']
new_df1.to_excel('train.xlsx',index=False)
new_df2 = new_df[new_df['训练集区分'] == '测试集']
new_df2.to_excel('test.xlsx',index=False)



