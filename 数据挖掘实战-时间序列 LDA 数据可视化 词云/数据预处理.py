import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import os
from tqdm import tqdm
from snownlp import SnowNLP


# 设置存储所有数据的空 DataFrame
df = pd.DataFrame()
# 遍历文件夹中的所有 csv 文件
folder_path = '婚俗改革'
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # 读取每个 csv 文件并合并到总的 DataFrame 中
        file_path = os.path.join(folder_path, file_name)
        try:
            data = pd.read_csv(file_path, encoding='gbk')
        except:
            data = pd.read_csv(file_path, encoding='utf-8-sig')
        df = pd.concat([df, data], ignore_index=True)


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


def sentiment_score(text):
    s = SnowNLP(text)
    sentiment = s.sentiments
    return round(sentiment,1)

print('原表格数据：',len(df))
df = df.drop_duplicates(subset=['全文内容'])
print('去重表格数据：',len(df))
df['全文内容'] = df['全文内容'].apply(preprocess_word)
df['全文内容'] = df['全文内容'].apply(emjio_tihuan)
df = df.dropna(subset=['全文内容'], axis=0)
df['fenci'] = df['全文内容'].apply(get_cut_words)
new_df = df.dropna(subset=['fenci'], axis=0)
print('数据预处理后表格数据：',len(new_df))
new_df['发帖数量'] = 1
new_df['sentiment score'] = new_df['fenci'].apply(sentiment_score)
new_df.to_csv('data.csv',index=False,encoding='utf-8-sig')


