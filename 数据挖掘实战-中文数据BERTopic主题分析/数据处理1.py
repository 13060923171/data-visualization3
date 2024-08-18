import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
from tqdm import tqdm

#使用自定义词典
jieba.load_userdict("custom_dict.txt")

# 导入停用词列表
stop_words = []
with open("Chinese stop words.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


#去掉标点符号，以及机械压缩
def preprocess_word(word):
    word1 = str(word)
    word1 = re.sub(r'转发微博', '', word1)
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
            if flag in ['a', 'an', 'n', 'nt', 'nz','v','vn']:
                if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                    # 如果是名词或形容词，就将其保存到列表中
                    nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN


df1 = pd.read_excel('data_v5.xlsx')
df1['清洗后内容'] = df1['清洗后内容'].apply(preprocess_word)
df1['清洗后内容'] = df1['清洗后内容'].apply(emjio_tihuan)
df1 = df1.dropna(subset=['清洗后内容'], axis=0)
df1['分词'] = df1['清洗后内容'].apply(get_cut_words)
df1 = df1.dropna(subset=['分词'], axis=0)
df1.to_excel('data_v6.xlsx',index=False)

