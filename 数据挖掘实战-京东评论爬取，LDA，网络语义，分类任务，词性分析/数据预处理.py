import pandas as pd
import numpy as np
import re
import jieba

df = pd.read_csv('data.csv')

def main1(x):
    x1 = str(x)
    if '快递' in x1 or '物流' in x1 or '送货' in x1 or '运输' in x1:
        return x1
    else:
        return np.NAN


df['物流信息'] = df['content'].apply(main1)
new_df = df.dropna(how='any',axis=0)

stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


def emjio_tihuan(x):
    x1 = str(x)
    x2 = re.sub('(\[.*?\])', "", x1)
    x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
    x4 = re.sub(r'\n', '', x3)
    return x4


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
    # 读入停用词表
    # 分词
    word_num = jieba.lcut(content_series, cut_all=False)

    # 条件筛选
    word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

    return ' '.join(word_num_selected)


new_df['物流信息'] = new_df['物流信息'].apply(emjio_tihuan)
new_df = new_df.dropna(subset=['物流信息'], axis=0)
new_df['物流信息'] = new_df['物流信息'].apply(yasuo)
new_df['分词'] = new_df['物流信息'].apply(get_cut_words)
new_df.to_csv('new_data.csv', encoding='utf-8-sig', index=False)

