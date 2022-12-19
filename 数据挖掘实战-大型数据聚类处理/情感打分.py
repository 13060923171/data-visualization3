import pandas as pd
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import numpy as np
import jieba


def snownlp_fx():
    df = pd.read_excel('demo-最终版.xlsx')
    df['评论'] = df['评论'].drop_duplicates(keep='first')
    df = df.dropna(subset=['评论'], axis=0)
    df['评论'] = df['评论'].astype(str)

    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    def get_cut_words(content_series):
        # 读入停用词表
        stop_words = []

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        # 分词
        word_num = jieba.lcut(content_series, cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

        if len(word_num_selected) != 0:
            score = SnowNLP(' '.join(word_num_selected))
            fenshu = score.sentiments
            return fenshu
        else:
            return None

    df['评分'] = df['评论'].apply(get_cut_words)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 6))
    plt.hist(df['评分'], bins=np.arange(0, 1, 0.01), facecolor='#E74C3C')
    plt.xlabel('情感数值')
    plt.ylabel('数量')
    plt.title('情感分析')
    plt.savefig('情感分析.jpg')
    plt.show()

    df.to_excel('情感分析demo.xlsx')

if __name__ == '__main__':
    snownlp_fx()

