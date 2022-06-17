import pandas as pd
# 数据处理库
import stylecloud
import jieba
import jieba.analyse
from IPython.display import Image


def wordclound_fx():
    df = pd.read_csv('公知.csv')

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

        # new_stop_words = []
        # with open("停用词1.txt", 'r', encoding='utf-8') as f:
        #     lines = f.readlines()
        # lines = lines[0].split('、')
        # for line in lines:
        #     new_stop_words.append(line.strip())
        # stop_words.extend(new_stop_words)

        # 分词
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected

    text1 = get_cut_words(content_series=df['4'])
    stylecloud.gen_stylecloud(text=' '.join(text1), max_words=100,
                              collocations=False,
                              font_path='simhei.ttf',
                              icon_name='fas fa-circle',
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              output_name='热词-词云图.png')
    Image(filename='热词-词云图.png')


    counts = {}
    for t in text1:
        counts[t] = counts.get(t, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('高频词.csv', encoding="utf-8-sig")


if __name__ == '__main__':
    wordclound_fx()