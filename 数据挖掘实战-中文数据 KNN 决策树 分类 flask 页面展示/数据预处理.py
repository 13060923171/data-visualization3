import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import os

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


def is_all_chinese_or_english(strs):
    for _char in strs:
        # 如果字符是中文或者英文，继续检查下一个字符
        if '\u4e00' <= _char <= '\u9fa5' or '\u0041' <= _char <= '\u005a' or '\u0061' <= _char <= '\u007a':
            continue
        # 如果字符既不是中文也不是英文，返回False
        else:
            return False
    # 所有字符都是中文或者英文，返回True
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
            if word not in stop_words and len(word) >= 2 and is_all_chinese_or_english(word) == True:
                # 如果是名词或形容词，就将其保存到列表中
                nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN

def get_all_file_names(directory):
    return os.listdir(directory)

directory = 'temp'  # 将这里的字符串改为你的目录的路径
file_names = get_all_file_names(directory)

sum_df = []
for file_name in file_names:
    df = pd.read_csv('./temp/{}'.format(file_name))
    file_name = str(file_name).replace('.csv','')
    df['MBTI'] = '{}'.format(file_name)
    df['评论'] = df['评论'].apply(preprocess_word)
    df['评论'] = df['评论'].apply(emjio_tihuan)
    df['评论'] = df['评论'].apply(yasuo)
    df.dropna(subset=['评论'], axis=0,inplace=True)
    df['分词'] = df['评论'].apply(get_cut_words)
    new_df = df.dropna(subset=['分词'], axis=0)
    sum_df.append(new_df)

data = pd.concat(sum_df,axis=0)
data = data.drop_duplicates(subset=['评论'])
data.to_csv('data.csv',encoding='utf-8-sig',index=False)




