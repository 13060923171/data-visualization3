import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from IPython.display import Image
import stylecloud


def wordclound_fx():
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
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    text = get_cut_words(content_series=df['评论'])
    stylecloud.gen_stylecloud(text=' '.join(text), max_words=200,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-circle',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='评论-词云图.png')
    Image(filename='评论-词云图.png')


    counts = {}
    for t in text:
        counts[t] = counts.get(t, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[:200]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('评论-高频词TOP200.csv', encoding="utf-8-sig")


if __name__ == '__main__':
    wordclound_fx()