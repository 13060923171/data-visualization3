import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
from snownlp import SnowNLP

# # 去掉重复行以及空值
# df = pd.read_excel('爬虫一年数据+2022.6.7-2023.6.7.xlsx',sheet_name="前半年（2022.6.7-2022.12.6）276629")
#
# # 导入停用词列表
# stop_words = []
# with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         stop_words.append(line.strip())
#
#
# #去掉标点符号，以及机械压缩
# def preprocess_word(word):
#     word1 = str(word)
#     # word1 = re.sub(r'转发微博', '', word1)
#     word1 = re.sub(r'#\w+#', '', word1)
#     word1 = re.sub(r'【.*?】', '', word1)
#     word1 = re.sub(r'@[\w]+', '', word1)
#     word1 = re.sub(r'[a-zA-Z]', '', word1)
#     word1 = re.sub(r'\.\d+', '', word1)
#     return word1
#
#
# def emjio_tihuan(x):
#     x1 = str(x)
#     x2 = re.sub('(\[.*?\])', "", x1)
#     x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
#     x4 = re.sub(r'\n', '', x3)
#     return x4
#
#
# # 判断是否为中文
# def is_all_chinese(strs):
#     for _char in strs:
#         if not '\u4e00' <= _char <= '\u9fa5':
#             return False
#     return True
#
#
# # 定义机械压缩函数
# def yasuo(st):
#     for i in range(1, int(len(st) / 2) + 1):
#         for j in range(len(st)):
#             if st[j:j + i] == st[j + i:j + 2 * i]:
#                 k = j + i
#                 while st[k:k + i] == st[k + i:k + 2 * i] and k < len(st):
#                     k = k + i
#                 st = st[:j] + st[k:]
#     return st
#
#
# def get_cut_words(content_series):
#     try:
#         # 对文本进行分词和词性标注
#         words = pseg.cut(content_series)
#         # 保存名词和形容词的列表
#         nouns_and_adjs = []
#         # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
#         for word, flag in words:
#             if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
#                 # 如果是名词或形容词，就将其保存到列表中
#                 nouns_and_adjs.append(word)
#         if len(nouns_and_adjs) != 0:
#             return ' '.join(nouns_and_adjs)
#         else:
#             return np.NAN
#     except:
#         return np.NAN
#
#
# def analyze_sentiment(text):
#     s = SnowNLP(text)
#     sentiment = s.sentiments
#     if sentiment >= 0.55:
#         return '正面情感'
#     else:
#         return '负面情感'
#
#
# df['text'] = df['text'].apply(preprocess_word)
# df['text'] = df['text'].apply(emjio_tihuan)
# df.dropna(subset=['text'], axis=0,inplace=True)
# # df['text'] = df['text'].apply(yasuo)
# df['分词'] = df['text'].apply(get_cut_words)
# new_df = df.dropna(subset=['分词'], axis=0)
# new_df = new_df.drop_duplicates(subset=['分词'])
# new_df['情感分类'] = new_df['分词'].apply(analyze_sentiment)
# new_df.to_csv('前半年数据.csv', encoding='utf-8-sig', index=False)


def main1():
    df = pd.read_csv("前半年数据.csv")

    def analyze_sentiment(text):
        s = SnowNLP(text)
        sentiment = s.sentiments
        sentiment = round(sentiment, 4)
        if sentiment >= 0.6:
            return '正面情感 {}'.format(sentiment)
        elif sentiment <= 0.4:
            return '负面情感 {}'.format(sentiment)
        else:
            return '中立情感 {}'.format(sentiment)

    def demo1(x):
        x1 = str(x).split(" ")
        return x1[0]

    def demo2(x):
        x1 = str(x).split(" ")
        return x1[1]

    df['情感属性'] = df['分词'].apply(analyze_sentiment)
    df['情感分类'] = df['情感属性'].apply(demo1)
    df['情感分值'] = df['情感属性'].apply(demo2)
    df = df.drop(['情感属性'],axis=1)
    df.to_csv('前半年数据.csv', encoding='utf-8-sig', index=False)


def main2():
    df = pd.read_csv("后半年数据.csv")

    def analyze_sentiment(text):
        s = SnowNLP(text)
        sentiment = s.sentiments
        sentiment = round(sentiment, 4)
        if sentiment >= 0.6:
            return '正面情感 {}'.format(sentiment)
        elif sentiment <= 0.4:
            return '负面情感 {}'.format(sentiment)
        else:
            return '中立情感 {}'.format(sentiment)

    def demo1(x):
        x1 = str(x).split(" ")
        return x1[0]

    def demo2(x):
        x1 = str(x).split(" ")
        return x1[1]

    df['情感属性'] = df['分词'].apply(analyze_sentiment)
    df['情感分类'] = df['情感属性'].apply(demo1)
    df['情感分值'] = df['情感属性'].apply(demo2)
    df = df.drop(['情感属性'], axis=1)
    df.to_csv('后半年数据.csv', encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    main1()
    main2()
