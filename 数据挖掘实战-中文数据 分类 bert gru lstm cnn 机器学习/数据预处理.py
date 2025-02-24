import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import os
from tqdm import tqdm


# def data_process():
#     df = pd.read_excel('./train/训练集合并-增加标签0220.xlsx')
#     data = pd.DataFrame()
#     data['评论内容'] = df['评论内容']
#     data['class'] = df['训练样本分类']
#
#     # 定义需要匹配的关键词列表
#     keywords = ['相对剥夺', '污名化', '其他', '群体对立', '不良示范']
#     # 构建正则表达式模式
#     pattern = r'(' + '|'.join(re.escape(kw) for kw in keywords) + r')'  # 防止特殊字符干扰
#
#     # 提取关键词并展开为多行
#     data['class'] = data['class'].apply(lambda x: re.findall(pattern, x))
#     data = data.explode('class').reset_index(drop=True)  # 展开列表为多行
#
#     data = data.drop_duplicates(subset=['评论内容'])
#     data.to_excel('./train/train.xlsx', index=False)


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
            # #判断是否为名词或者形容词或者动词
            if flag in ['Ag','a','ad','an','Ng','n','v']:
                if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                        # 如果是名词或形容词，就将其保存到列表中
                        nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN


if __name__ == '__main__':
    # data_process()
    df = pd.read_excel('./train/测试集合并-减少标签0220.xlsx')
    df = df.drop_duplicates(subset=['评论内容'])
    print('原数据总数:', len(df))
    df['评论内容'] = df['评论内容'].apply(preprocess_word)
    df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
    df = df.dropna(subset=['评论内容'], axis=0)
    df['fenci'] = df['评论内容'].apply(get_cut_words)
    df = df.dropna(subset=['fenci'], axis=0)
    print('清洗过后数据总数:', len(df))
    df.to_excel('./test/预测文本.xlsx', index=False)