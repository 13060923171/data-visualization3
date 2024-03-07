import jieba
import jieba.posseg as pseg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import re
from collections import Counter

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
            if word not in stop_words and is_all_chinese(word) == True:
                # 如果是名词或形容词，就将其保存到列表中
                nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN


def top10_keyword(content_series):
    line1 = str(content_series).split(' ')
    # 计算关键词
    c = Counter()
    for x in line1:
        c[x] += 1
    output = ""
    for (k, v) in c.most_common(10):
        output += k + " "

    return output


def jieba_precision():
    # # 使用Jieba分词
    jieba_result = [pseg.cut(d) for d in new_df['fenci']]
    jieba_result = [(word.word, word.flag) for sublist in jieba_result for word in sublist]
    jieba_result1 = []
    list_true = []
    for word,flag in jieba_result:
        if is_all_chinese(word) == True:
            true_flag = word_dict.get(word)
            if true_flag is not None:
                list_true.append(true_flag)
                jieba_result1.append(flag)

    jr_prec, jr_rec, jr_f_score,_ = precision_recall_fscore_support(list_true, jieba_result1, average='weighted')
    print("Jieba: Precision: {:.2f}, Recall: {:.2f}, FScore: {:.2f}".format(jr_prec, jr_rec, jr_f_score))
    data = pd.DataFrame()
    data['Precision'] = ['{:.2f}'.format(jr_prec)]
    data['Recall'] = ['{:.2f}'.format(jr_rec)]
    data['FScore'] = ['{:.2f}'.format(jr_f_score)]
    data.to_csv('prec_jieba.csv',index=False,encoding='utf-8-sig')


if __name__ == '__main__':
    # 去掉重复行以及空值
    df = pd.read_csv('评论内容.csv')
    df = df.drop_duplicates(subset=['评论内容'])
    df['评论内容'] = df['评论内容'].apply(preprocess_word)
    df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
    df = df.dropna(subset=['评论内容'], axis=0)
    df['fenci'] = df['评论内容'].apply(get_cut_words)
    new_df = df.dropna(subset=['fenci'], axis=0)
    new_df['keyword'] = new_df['fenci'].apply(top10_keyword)
    new_df.to_csv('data.csv', index=False, encoding='utf-8-sig')
    new_df.to_csv('jieba_data.csv',index=False,encoding='utf-8-sig')

    clean_dict = {}
    with open('文本语料库.txt', 'r', encoding='utf-8-sig') as f:
        corpus = f.readlines()
    for line in corpus:
        items = line.split()
        for item in items:
            parts = item.rsplit('/', 1)
            if len(parts) > 1:
                key, value = parts[0], parts[1]
                if key not in clean_dict and is_all_chinese(key) == True:
                    clean_dict[key] = value

    word_dict = {}
    for key, value in clean_dict.items():
        if value not in word_dict.values():
            word_dict[key] = value

    jieba_precision()


