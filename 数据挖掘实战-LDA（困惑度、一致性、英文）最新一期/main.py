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


#LDA建模
def lda():
    #读取数据 删除重复项，空值
    df = pd.read_excel('LDA分析.xlsx')
    df = df.dropna(subset=['text'], axis=0)
    content = df['text'].drop_duplicates(keep='first')
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

    # 构建同义词典
    def wordnet_lemmas(token):
        try:
            synonyms = []
            for syn in wordnet.synsets(token):
                for lm in syn.lemmas():
                    synonyms.append(lm.name())
            synonym = list(set(synonyms))
            synonym.sort(key=lambda x: x, reverse=False)
            return synonym[0]
        except:
            return token

    f = open('class-fenci.txt', 'w', encoding='utf-8-sig')
    for c in content:
        #文本分词
        tokens = [word.lower() for sent in nltk.sent_tokenize(c) for word in nltk.word_tokenize(sent)]
        # 计算关键词
        c = Counter()
        filtered_tokens = []
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
                if token not in stop_words and len(token) >= 4:
                    c[token] += 1
        # Top30
        output = ""
        for (k, v) in c.most_common(30):
            output += k + " "
        f.write(output + "\n")
    else:
        f.close()


    fr = open('class-fenci.txt', 'r', encoding='utf-8-sig')
    train = []
    d = {}
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ') if len(word) >= 2]
        for l in line:
            d[l] = d.get(l,0)+1
        train.append(line)
    ls = list(d.items())
    ls.sort(key=lambda x:x[1],reverse=True)

    x1 = []
    y1 = []
    for key,values in ls[0:200]:
        x1.append(key)
        y1.append(values)

    data = pd.DataFrame()
    data['word'] = x1
    data['counts'] = y1
    data.to_csv('高频词.csv',encoding="utf-8-sig",index=False)

    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    #困惑度模块
    x_data = []
    y_data = []
    z_data = []
    for i in tqdm(range(2,15)):
        x_data.append(i)
        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, random_state=111, iterations=400)
        #困惑度计算
        perplexity = lda.log_perplexity(corpus)
        y_data.append(perplexity)
        #一致性计算
        coherencemodel = models.CoherenceModel(model=lda, texts=train, dictionary=dictionary, coherence='c_v')
        z_data.append(coherencemodel.get_coherence())

    # 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 绘制困惑度折线图
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x_data, y_data, marker="o")
    plt.title("perplexity_values")
    plt.xlabel('num topics')
    plt.ylabel('perplexity score')


    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x_data, z_data, marker="o")
    plt.title("coherence_values")
    plt.xlabel("num topics")
    plt.ylabel("coherence score")

    plt.savefig('demo.png')
    plt.show()

    num_topics = input("请输入具体整体:")
    #LDA可视化模块
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
    data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(data1, 'lda.html')


if __name__ == '__main__':
    lda()
