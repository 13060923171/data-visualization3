import pandas as pd
# 数据处理库
import numpy as np
import re
from collections import Counter
from gensim import corpora, models
import itertools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import matplotlib
import nltk
from nltk.stem.porter import PorterStemmer #词干提取
from nltk.stem import WordNetLemmatizer    #词性还原
from nltk.corpus import wordnet #构建同义词典
from nltk import word_tokenize, pos_tag #词性标注，分词
from tqdm import tqdm
import string
from transformers import pipeline

#LDA建模
def lda():
    #读取数据 删除重复项，空值
    df = pd.read_excel('【处理后】评论数据.xlsx')
    df = df.dropna(subset=['content'], axis=0)
    content = df['content'].drop_duplicates(keep='first')
    content = content.dropna(how='any')

    #添加停用词
    stop_words = []
    with open("常用英文停用词(NLP处理英文必备)stopwords.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    # 获取单词的词性。词干提取
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    #词性还原
    lemmatizer = WordNetLemmatizer()

    f = open('fenci.txt', 'w', encoding='utf-8-sig')
    for c in content:
        # #文本分词
        tokens = [word.lower() for sent in nltk.sent_tokenize(c) for word in nltk.word_tokenize(sent)]
        # 计算关键词
        c = Counter()
        # 过滤所有不含字母的词例（例如：数字、纯标点）
        tagged_sent = pos_tag(tokens)
        lemmas_sent = []
        for tag in tagged_sent:
            #词性标注
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(lemmatizer.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原

        for token in lemmas_sent:
            if re.search('[a-zA-Z]', token):
                # token = wordnet_lemmas(token)
                if token not in stop_words and len(token) >= 3:
                    c[token] += 1
        # Top30
        output = ""
        for (k, v) in c.most_common(30):
            output += k + " "
        f.write(output + "\n")
    else:
        f.close()

    fr = open('fenci.txt', 'r', encoding='utf-8-sig')
    train = []
    d = {}
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ') if len(word) >= 3]
        for l in line:
            d[l] = d.get(l,0)+1
        train.append(line)


    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    num_topics = 6
    #LDA可视化模块
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)

    # 主题判断模块
    list3 = []
    list2 = []
    for i in lda.get_document_topics(corpus)[:]:
        listj = []
        list1 = []
        for j in i:
            list1.append(j)
            listj.append(j[1])
        list3.append(list1)
        bz = listj.index(max(listj))
        list2.append(i[bz][0])

    df['主题概率'] = list3
    df['主题类型'] = list2

    classifier = pipeline('sentiment-analysis')
    label_list = []
    score_list = []
    for d in df['content']:
        class1 = classifier(d)
        label = class1[0]['label']
        score = class1[0]['score']
        if score <= 0.6:
            label = 'NEUTRAL'
            label_list.append(label)
        else:
            label = label
            label_list.append(label)
        score_list.append(score)

    df['情感类型'] = label_list
    df['情感得分'] = score_list
    df.to_csv('new_data.csv', encoding='utf-8-sig', index=False)



def main1():
    df = pd.read_csv('new_data.csv')
    def demo(x):
        df1 = x
        new_df1 = df1['情感类型'].value_counts()
        new_df1 = new_df1.sort_index()
        return new_df1

    new_df = df.groupby('主题类型').apply(demo)
    new_df['sum'] = new_df['NEGATIVE'] + new_df['NEUTRAL'] + new_df['POSITIVE']
    new_df['NEGATIVE'] = round(new_df['NEGATIVE'] / new_df['sum'],4)
    new_df['NEUTRAL'] = round(new_df['NEUTRAL'] / new_df['sum'], 4)
    new_df['POSITIVE'] = round(new_df['POSITIVE'] / new_df['sum'], 4)
    new_df = new_df.drop(['sum'],axis=1)
    new_df.to_excel('结果表格.xlsx')


if __name__ == '__main__':
    # lda()
    main1()
