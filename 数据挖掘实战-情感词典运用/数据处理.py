import jieba
import jieba.analyse
import pandas as pd
import re

df1 = pd.read_csv('data.csv')
df2 = pd.read_csv('data1.csv',encoding='gb18030')
data = pd.concat([df1,df2],axis=0)
data = data.drop_duplicates(keep='first')
data = data.dropna(subset=['txt_paragraph'], axis=0)


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

def get_cut_words(str1):

    # 读入停用词表
    stop_words = []

    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    # 分词
    word_num = jieba.lcut(str(str1), cut_all=False)

    # 条件筛选
    word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

    word_num_selected = " ".join(word_num_selected)

    return word_num_selected


data['txt_paragraph'] = data['txt_paragraph'].apply(emjio_tihuan)
data = data.dropna(subset=['txt_paragraph'], axis=0)
data['participle_process'] = data['txt_paragraph'].apply(get_cut_words)
data.to_csv('sum_data.csv',encoding='utf-8-sig',index=False)


# import pandas as pd
# import numpy as np
# df = pd.read_csv('new_sum_data.csv',encoding='utf-8-sig')
# df['sentiments_score'] = df['sentiments_score'].replace(0,np.NAN)
# df = df.dropna(how='any',axis=0)
# df.to_csv('sum_data.csv',encoding='utf-8-sig')