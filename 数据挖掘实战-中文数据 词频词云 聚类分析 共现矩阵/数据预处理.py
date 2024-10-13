import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import os
from tqdm import tqdm


# # 导入停用词列表
# stop_words = []
# with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         stop_words.append(line.strip())
def main1(name1,name2,name3):
    word1s = []
    data = pd.read_excel(f'{name1}.xlsx')
    for d in data['top20']:
        word1s.append(d)

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
                # if flag in ['Ag','a','ad','an','Ng','n','nr','ns','nt','nz']:
                # 判断是否为名词或者形容词或者动词
                if flag in ['Ag','a','ad','an','Ng','n','v']:
                    if word in word1s and len(word) >= 2 and is_all_chinese(word) == True:
                        # 如果是名词或形容词，就将其保存到列表中
                        nouns_and_adjs.append(word)
            if len(nouns_and_adjs) != 0:
                return ' '.join(nouns_and_adjs)
            else:
                return np.NAN
        except:
            return np.NAN




    df = pd.read_excel(f'{name2}.xlsx')
    df = df.drop_duplicates(subset=['内容'])
    df['内容'] = df['内容'].apply(preprocess_word)
    df['内容'] = df['内容'].apply(emjio_tihuan)
    df = df.dropna(subset=['内容'], axis=0)
    df['fenci'] = df['内容'].apply(get_cut_words)
    new_df = df.dropna(subset=['fenci'], axis=0)
    new_df.to_csv(f'{name3}.csv',index=False,encoding='utf-8-sig')


if __name__ == '__main__':
    list_name1 = ['（1）高频词Top200','（2）高频词Top200','（3）高频词Top200']
    list_name2 = ['1994～1999 年','2000～2011 年','2012～2024 年']
    list_name3 = ['data1','data2','data3']
    for n1,n2,n3 in zip(list_name1,list_name2,list_name3):
        main1(n1,n2,n3)

