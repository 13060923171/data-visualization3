import re
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from collections import Counter
import jieba.posseg as posseg

df = pd.read_excel('评论信息.xlsx')
df = df.drop_duplicates()

# df = pd.read_csv('new_data.csv')
# df = df.drop_duplicates()
# df = df[df['分类结果'] == '负面评论']

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

stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())

df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
df = df.dropna(subset=['评论内容'], axis=0)


def fenci():
    f = open('fenci.txt', 'w', encoding='utf-8-sig')
    for line in df['评论内容']:
        line = line.strip('\n')
        # 停用词过滤
        seg_list = jieba.cut(line, cut_all=False)
        cut_words = (" ".join(seg_list))

        # 计算关键词
        all_words = cut_words.split()
        c = Counter()
        for x in all_words:
            if len(x) >= 2 and x != '\r\n' and x != '\n':
                if is_all_chinese(x) == True and x not in stop_words:
                    c[x] += 1
        output = ""
        for (k, v) in c.most_common(30):
            output += k + " "

        f.write(output + "\n")
    else:
        f.close()


def td_idf():
    corpus = []
    # 读取预料 一行预料为一个文档
    for line in open('fenci.txt', 'r', encoding='utf-8-sig').readlines():
        corpus.append(line.strip())

        # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names_out()

    # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    data = {'word': word,
            'tfidf': weight.sum(axis=0).tolist()}

    df2 = pd.DataFrame(data)
    df2['tfidf'] = df2['tfidf'].astype('float64')
    df2 = df2.sort_values(by=['tfidf'], ascending=False)
    df3 = df2.iloc[:100]
    df3.to_csv('Top100权重词.csv',encoding='utf-8-sig',index=False)


def word_emotion():
    df = pd.read_csv('new_data.csv')
    df = df.drop_duplicates()
    df = df[df['分类结果'] == '非负评论']
    df['评论内容'] = df['评论内容'].apply(emjio_tihuan)
    df = df.dropna(subset=['评论内容'], axis=0)

    counts = {}
    for d in df['评论内容']:
        res = posseg.cut(d)
        for word, flag in res:
            if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                if flag == 'Ag' or flag == 'a' or flag == 'ad' or flag == 'an':
                    counts[word] = counts.get(word, 0) + 1

    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []

    for key, values in ls[:100]:
        x_data.append(key)
        y_data.append(values)

    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('Top100_非负形容词.csv', encoding="utf-8-sig")


if __name__ == '__main__':
    # fenci()
    # td_idf()
    word_emotion()





