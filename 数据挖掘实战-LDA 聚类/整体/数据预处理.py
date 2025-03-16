import pandas as pd
import numpy as np
import re
import jieba
import jieba.analyse
import jieba.posseg as pseg
import os
from tqdm import tqdm

# data = pd.read_excel('./数据/初中-Top100关键词.xlsx')
#
#
# with open('custom_dict1.txt', 'w', encoding='utf-8-sig') as f:
#     for text in data['word']:
#         f.write(text + '\n')  # 自动添加换行符
#
#
# 读取第一个文件并处理换行符
with open('custom_dict.txt', 'r', encoding='utf-8-sig') as f:
    words_custom = [line.strip() for line in f]

# # 读取第二个文件并处理换行符
# with open('custom_dict1.txt', 'r', encoding='utf-8-sig') as f:
#     words_custom1 = [line.strip() for line in f]


combined_words =  words_custom
# 使用字典键的顺序保持唯一性并去重
combined_words = list(dict.fromkeys(combined_words))

# # 将合并后的内容写入新文件
# with open('merged_dict.txt', 'w', encoding='utf-8-sig') as f:
#     for word in combined_words:
#         f.write(f"{word}\n")


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
            # if flag in ['Ag','a','ad','an','Ng','n','nz','nr','ns','nt','nz','v','vn']:
                if word not in stop_words and len(word) >= 2 and word in words_custom:
                    # 如果是名词或形容词，就将其保存到列表中
                    nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN


df = pd.read_excel('./数据/初中-总数据.xlsx')
print('原数据总数:',len(df))
df['内容'] = df['内容'].apply(preprocess_word)
df['内容'] = df['内容'].apply(emjio_tihuan)
df = df.dropna(subset=['内容'], axis=0)
df['fenci'] = df['内容'].apply(get_cut_words)
df = df.dropna(subset=['fenci'], axis=0)
print('清洗过后数据总数:',len(df))
df.to_excel('初中数据.xlsx',index=False)




