import pandas as pd
import nltk
import re
from collections import Counter
from nltk.stem.snowball import SnowballStemmer  # 返回词语的原型，去掉ing等
stemmer = SnowballStemmer("english")


stop_words = []
with open("常用英文停用词(NLP处理英文必备)stopwords.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


def tokenize_only(text):  # 分词器，仅分词
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return ' '.join(filtered_tokens)


def main1():
    # 绘制词云图
    df1 = pd.read_csv('new_temp.csv')
    df = df1[df1['translate'] == 'en']
    df['new_content'] = df['new_content'].apply(tokenize_only)
    df['new_content'].astype('str')
    f = open('C-class-fenci.txt', 'w', encoding='utf-8')
    for d in df['new_content']:
        c = Counter()
        tokens = nltk.word_tokenize(d)
        tagged = nltk.pos_tag(tokens)
        for t in tagged:
            if t[1] in "NN" and t[0] not in stop_words and t[0] != '\r\n' and len(t[0]) > 1:
                c[t[0]] += 1
        # Top20
        output = ""
        for (k, v) in c.most_common(20):
            # print("%s:%d"%(k,v))
            output += k + " "
        f.write(output + "\n")

    else:
        f.close()
    #
    #
    #
    # counts = {}
    # for s in sum_cotent:
    #     counts[s] = counts.get(s,0)+1
    # ls = list(counts.items())
    # ls.sort(key=lambda x:x[1],reverse=True)
    # x_data = []
    # y_data = []
    # for key,values in ls[:1000]:
    #     x_data.append(key)
    #     y_data.append(values)

    # df1 = pd.DataFrame()
    # df1['word'] = x_data
    # df1['counts'] = y_data
    # df1.to_csv('top1000-高频词-名词.csv',encoding='utf-8-sig')


if __name__ == '__main__':
    main1()
