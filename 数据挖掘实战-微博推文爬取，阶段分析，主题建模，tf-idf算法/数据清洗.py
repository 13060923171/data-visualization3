import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg

df = pd.read_csv('钟薛高.csv')
df.drop_duplicates(subset=['内容'], inplace=True)


def time_process(x):
    if '今天' in x:
        return np.NaN
    else:
        return x

df['时间'] = df['时间'].apply(time_process)
df.dropna(subset=['时间'],axis=0, inplace=True)

# 导入停用词列表
stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def is_all_letters(text):
    return text.isalpha()

def is_all_digits(text):
    return text.isdigit()

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


#去掉标点符号，以及机械压缩
def preprocess_word(word):
    word = re.sub(r'转发微博', '', word)
    word = re.sub(r'#\w+#', '', word)
    word = re.sub(r'@[\w]+', '', word)
    word = re.sub(r'[a-zA-Z]', '', word)
    word = re.sub(r'\.\d+', '', word)
    return word


def get_cut_words(content_series):
    jieba.load_userdict("user_dict.txt")
    # jieba.add_word(["钟薛高",'一个小时','1小时'])  # 添加特定词汇
    # 对文本进行分词和词性标注
    words = pseg.cut(content_series)
    # 保存名词和形容词的列表
    nouns_and_adjs = []
    # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
    for word, flag in words:
        if word not in stop_words and len(word) >= 2 and is_all_digits(word) == False:
            nouns_and_adjs.append(word)
    if len(nouns_and_adjs) != 0:
        return ' '.join(nouns_and_adjs)
    else:
        return np.NAN

df['内容'] = df['内容'].apply(preprocess_word)
df['内容'] = df['内容'].apply(yasuo)
df['分词'] = df['内容'].apply(get_cut_words)
new_df = df.dropna(subset=['分词'], axis=0)
new_df.to_csv('new_data.csv', encoding='utf-8-sig', index=False)
