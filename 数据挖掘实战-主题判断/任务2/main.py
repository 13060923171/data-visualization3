import pandas as pd
# 数据处理库
import numpy as np
import re
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
from gensim import corpora, models
import itertools
import pyLDAvis
import pyLDAvis.gensim


def lda_tfidf():
    df = pd.read_csv('公知.csv')
    df = df.dropna(how='any',axis=0)
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    stop_words = []

    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    # new_stop_words = []
    # with open("停用词2.txt", 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    # lines = lines[0].split('、')
    # for line in lines:
    #     new_stop_words.append(line.strip())
    # stop_words.extend(new_stop_words)

    # jieba.analyse.set_stop_words(stop_words)
    f = open('class-fenci.txt', 'w', encoding='utf-8')
    for line in df['4']:
        line = line.strip('\n')
        # 停用词过滤
        line = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", line)
        seg_list = jieba.cut(line, cut_all=False)
        cut_words = (" ".join(seg_list))

        # 计算关键词
        all_words = cut_words.split()
        c = Counter()
        for x in all_words:
            if len(x) > 1 and x != '\r\n':
                if is_all_chinese(x) == True and x not in stop_words:
                    c[x] += 1
        # Top50
        output = ""
        # print('\n词频统计结果：')
        for (k, v) in c.most_common():
            # print("%s:%d"%(k,v))
            output += k + " "

        f.write(output + "\n")
    else:
        f.close()

    # fr = open('class-fenci.txt', 'r', encoding='utf-8')
    # train = []
    #
    # for line in fr.readlines():
    #     line = [word for word in line.split(' ') if word != '\n']
    #     train.append(line)
    #
    # dictionary = corpora.Dictionary(train)
    # dictionary.filter_extremes(no_below=2, no_above=1.0)
    # corpus = [dictionary.doc2bow(text) for text in train]
    #
    # # 构造主题数寻优函数
    # def cos(vector1, vector2):  # 余弦相似度函数
    #     dot_product = 0.0
    #     normA = 0.0
    #     normB = 0.0
    #     for a, b in zip(vector1, vector2):
    #         dot_product += a * b
    #         normA += a ** 2
    #         normB += b ** 2
    #     if normA == 0.0 or normB == 0.0:
    #         return (None)
    #     else:
    #         return (dot_product / ((normA * normB) ** 0.5))
    #
    #     # 主题数寻优
    #
    # def lda_k(x_corpus, x_dict):
    #     # 初始化平均余弦相似度
    #     mean_similarity = []
    #     mean_similarity.append(1)
    #
    #     # 循环生成主题并计算主题间相似度
    #     for i in np.arange(2, 11):
    #         lda = models.LdaModel(x_corpus, num_topics=i, id2word=x_dict)  # LDA模型训练
    #         for j in np.arange(i):
    #             term = lda.show_topics(num_words=50)
    #
    #         # 提取各主题词
    #         top_word = []
    #         for k in np.arange(i):
    #             top_word.append([''.join(re.findall('"(.*)"', i)) \
    #                              for i in term[k][1].split('+')])  # 列出所有词
    #
    #         # 构造词频向量
    #         word = sum(top_word, [])  # 列出所有的词
    #         unique_word = set(word)  # 去除重复的词
    #
    #         # 构造主题词列表，行表示主题号，列表示各主题词
    #         mat = []
    #         for j in np.arange(i):
    #             top_w = top_word[j]
    #             mat.append(tuple([top_w.count(k) for k in unique_word]))
    #
    #         p = list(itertools.permutations(list(np.arange(i)), 2))
    #         l = len(p)
    #         top_similarity = [0]
    #         for w in np.arange(l):
    #             vector1 = mat[p[w][0]]
    #             vector2 = mat[p[w][1]]
    #             top_similarity.append(cos(vector1, vector2))
    #
    #         # 计算平均余弦相似度
    #         mean_similarity.append(sum(top_similarity) / l)
    #     return (mean_similarity)

    # # 计算主题平均余弦相似度
    # word_k = lda_k(corpus, dictionary)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(10, 8), dpi=300)
    # plt.plot(word_k)
    # plt.title('LDA评论主题数寻优')
    # plt.xlabel('主题数')
    # plt.ylabel('平均余弦相似度')
    # plt.savefig('LDA评论主题数寻优.png')
    # plt.show()
    #
    # topic_lda = word_k.index(min(word_k)) + 1

    # lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=6)
    # data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    # pyLDAvis.save_html(data, 'lda.html')


if __name__ == '__main__':
    lda_tfidf()