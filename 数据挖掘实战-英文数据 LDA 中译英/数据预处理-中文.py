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
            # if flag in ['Ag','a','ad','an','Ng','n','nr','ns','nt','nz']:
            if flag in ['Ag','a','ad','an','Ng','n','nr','ns','nt','nz']:
                if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                    # 如果是名词或形容词，就将其保存到列表中
                    nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN


# df1 = pd.read_excel('./post/douy_post.xlsx')
# df2 = pd.read_excel('./post/xhs_post.xlsx')
# df3 = pd.read_excel('./post/zhihu_post.xlsx')
#
# data1 = pd.DataFrame()
# data1['作者id'] = df1['用户id']
# data1['标题'] = df1['标题']
# data2 = pd.DataFrame()
# data2['作者id'] = df2['作者id']
# data2['标题'] = df2['标题']
# data3 = pd.DataFrame()
# data3['作者id'] = df3['回答id']
# data3['标题'] = df3['回答内容']
#
# df = pd.concat([data1,data2,data3],axis=0)
#
# df = df.drop_duplicates(subset=['标题'])
# df['标题'] = df['标题'].apply(preprocess_word)
# df['标题'] = df['标题'].apply(emjio_tihuan)
# df = df.dropna(subset=['标题'], axis=0)
# df['分词'] = df['标题'].apply(get_cut_words)
# new_df = df.dropna(subset=['分词'], axis=0)
# new_df.to_excel('./post/zh_post.xlsx',index=False)


# df1 = pd.read_excel('./comment/dy_comen.xlsx')
# df2 = pd.read_excel('./comment/xhs_comen.xlsx')
df3 = pd.read_excel('./comment/zhihu_comen.xlsx')
# data1 = pd.DataFrame()
# data1['评论用户id'] = df1['评论用户id']
# data1['评论内容'] = df1['评论内容']
# data2 = pd.DataFrame()
# data2['评论用户id'] = df2['用户id']
# data2['评论内容'] = df2['评论正文']
data3 = pd.DataFrame()
data3['评论用户id'] = df3['评论id']
data3['评论内容'] = df3['评论内容']

df = pd.concat([data3],axis=0)

df = df.drop_duplicates(subset=['评论内容'])
df['评论内容'] = df['评论内容'].apply(preprocess_word)
df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
df = df.dropna(subset=['评论内容'], axis=0)
df['分词'] = df['评论内容'].apply(get_cut_words)
new_df = df.dropna(subset=['分词'], axis=0)
new_df.to_excel('./comment/zh_comment.xlsx',index=False)