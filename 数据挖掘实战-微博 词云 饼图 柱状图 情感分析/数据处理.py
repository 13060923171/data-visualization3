import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
from snownlp import SnowNLP

# 去掉重复行以及空值
df = pd.read_excel('筛选过后数据.xlsx')


# 导入停用词列表
stop_words = []
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
            if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                # 如果是名词或形容词，就将其保存到列表中
                nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN


def analyze_sentiment(text):
    s = SnowNLP(text)
    sentiment = s.sentiments
    if sentiment > 0.5:
        return '积极态度'
    elif sentiment < 0.5:
        return '消极态度'
    else:
        return '中立态度'

# keyword1 = '徽州雕刻'
# keyword2 = '雕刻'
# keyword3 = '徽州三雕'
# keyword4 = '黟县'
# keyword5 = '歙县'
# keyword6 = '休宁'
# keyword7 = '徽州'
# keyword8 = '婺源'
# keyword9 = '绩溪'
# keyword10 = '屯溪'
# keyword11 = '竹雕'
# keyword12 = '木雕'
# keyword13 = '砖雕'
# keyword14 = '石雕'
# keyword15 = '墨'
# keyword16 = '砚'
#
# new_df = df[(df['博文内容'].str.contains('{}'.format(keyword1), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword2), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword3), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword4), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword5), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword6), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword7), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword8), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword9), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword10), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword11), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword12), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword13), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword14), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword15), case=False)) |
#             (df['博文内容'].str.contains('{}'.format(keyword16), case=False))]


# df = df.drop_duplicates(subset=['博文内容'])
df = df.drop_duplicates(subset=['博文内容'])
# 获取你想要删除的列的名称
cols_to_drop = df.columns[17:]
# 使用 drop 函数删除列
df = df.drop(cols_to_drop, axis=1)

df['博文内容'] = df['博文内容'].apply(preprocess_word)
df['博文内容'] = df['博文内容'].apply(emjio_tihuan)
df.dropna(subset=['博文内容'], axis=0,inplace=True)
# df['评论'] = df['评论'].apply(yasuo)
df['博文内容_分词'] = df['博文内容'].apply(get_cut_words)
new_df = df.dropna(subset=['博文内容_分词'], axis=0)
new_df = new_df.drop_duplicates(subset=['博文内容_分词'])
new_df['情感分类_博文'] = new_df['博文内容_分词'].apply(analyze_sentiment)
new_df.to_excel('博文.xlsx', index=False)

