import pandas as pd
import numpy as np
import re
import jieba
import paddlehub as hub

df = pd.read_excel('微博数据.xlsx')

stop_words = ['抱歉','作者','设置','此微博','内微博','半年','展示','网页','微博','链接','视频','查看']
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
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


df['new_content'] = df['content'].apply(preprocess_word)
df['new_content'] = df['new_content'].apply(emjio_tihuan)
df.dropna(subset=['new_content'], axis=0,inplace=True)
df['new_content'] = df['new_content'].apply(yasuo)
df['new_content'] = df['new_content'].apply(get_cut_words)
df.dropna(subset=['content'], axis=0,inplace=True)
senta = hub.Module(name="senta_bilstm")
texts = df['content'].tolist()
input_data = {'text': texts}
res = senta.sentiment_classify(data=input_data)
df['情感分值'] = [x['positive_probs'] for x in res]
df.to_csv('new_data.csv', encoding='utf-8-sig', index=False)
