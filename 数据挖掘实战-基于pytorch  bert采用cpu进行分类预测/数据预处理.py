import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg

# 去掉重复行以及空值
df = pd.read_excel('data.xlsx')

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


def get_cut_words(post_title_series):
    try:
        # 对文本进行分词和词性标注
        words = pseg.cut(post_title_series)
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


df = df.drop_duplicates(subset=['post_title'])
df['post_title'] = df['post_title'].apply(preprocess_word)
df['post_title'] = df['post_title'].apply(emjio_tihuan)
df = df.dropna(subset=['post_title'], axis=0)
df['fenci'] = df['post_title'].apply(get_cut_words)
new_df = df.dropna(subset=['fenci'], axis=0)


def data_process(x):
    x1 = str(x)
    if x1 != 'nan':
        return '训练集'
    else:
        return '测试集'


df['数据集'] = df['情感分类'].apply(data_process)

df1 = df[df['数据集'] == '训练集']
df1.to_excel('train_data.xlsx',index=False)
df2 = df[df['数据集'] == '测试集']
df2.to_excel('test_data.xlsx',index=False)



