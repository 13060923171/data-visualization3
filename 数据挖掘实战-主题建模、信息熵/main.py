import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import time
from nltk.stem import WordNetLemmatizer
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import itertools
import pyLDAvis
import pyLDAvis.gensim
from tqdm import tqdm
import os
from gensim.models import LdaModel
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

df = pd.read_excel('BYDCompany发布.xlsx')
lemmatizer = WordNetLemmatizer()


stop_words = []
with open('常用英文停用词(NLP处理英文必备)stopwords.txt','r',encoding='utf-8')as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip().replace("'",""))

#去掉标点符号，以及机械压缩
def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word

#再次去掉标点符号
def gettext(x):
    import string
    punc = string.punctuation
    for ch in punc:
        txt = str(x).replace(ch,"")
    return txt


#替换表情包
def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' ', tweet)
    # Angry -- >:(, >:-(, :'(
    tweet = re.sub(r'(>:\(|>:-\(|:\'\()', ' ', tweet)
    # Surprised -- :O, :o, :-O, :-o, :0, 8-0
    tweet = re.sub(r'(:\s?[oO]|:-[oO]|:0|8-0)', ' ', tweet)
    # Confused -- :/, :\, :-/, :-\
    tweet = re.sub(r'(:\\|:/|:-\\\\|:-/)', ' ', tweet)
    # Embarrassed -- :$, :-$
    tweet = re.sub(r'(:\\$|:-\\$)', ' ', tweet)
    # Other emoticons
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', tweet)
    for emoticon in emoticons:
        tweet = tweet.replace(emoticon, " ")
    return tweet


def clean_text(text):
    # Replaces URLs with the word URL
    text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', text)
    # Replace @username with the word USER_MENTION
    text = re.sub(r'@[\S]+', ' ', text)
    # Replace #hashtag with the word HASHTAG
    text = re.sub(r'#(\S+)', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    # Remove RT (retweet)
    text = re.sub(r'\brt\b', ' ', text)
    # Replace 2+ dots with space
    text = re.sub(r'\.{2,}', ' ', text)
    # Strip space, " and ' from text
    text = text.strip(' "\'')
    # Handle emojis
    text = handle_emojis(text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
    # Lowercase and split into words
    words = text.lower().split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [preprocess_word(w) for w in words]
    if len(words) != 0:
        return ' '.join(words)
    else:
        return np.NAN


df['clearn_comment'] = df['内容1'].apply(gettext)
df['clearn_comment'] = df['clearn_comment'].apply(preprocess_word)
df['clearn_comment'] = df['clearn_comment'].apply(clean_text)
df.dropna(subset=['clearn_comment'],axis=0,inplace=True)

dictionary = corpora.Dictionary([text.split() for text in df['clearn_comment']])
corpus = [dictionary.doc2bow(text.split()) for text in df['clearn_comment']]


num_topics = input('请输入主题数:')

#LDA可视化模块
#构建lda主题参数
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
#读取lda对应的数据
data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
#把数据进行可视化处理
pyLDAvis.save_html(data1, 'lda.html')

# 计算主题的信息熵
topic_entropies = []
for topic in lda.show_topics():
    topic_words = [word for word, prob in lda.show_topic(topic[0])]
    word_probs = [prob for word, prob in lda.show_topic(topic[0])]
    entropy = np.sum(-np.array(word_probs) * np.log2(word_probs))
    topic_entropies.append((topic[0], topic_words, entropy))

z_data1 = []
x_data1 = []
y_data1 = []
# 打印主题及其对应的信息熵
for topic_entropy in topic_entropies:
    print("Topic {}: {}".format(topic_entropy[0], topic_entropy[1]))
    x_data1.append(topic_entropy[0])
    y_data1.append(topic_entropy[1])
    print("Entropy: {}".format(topic_entropy[2]))
    z_data1.append(topic_entropy[2])

#将上面获取的数据进行保存
df2 = pd.DataFrame()
df2['主题数'] = x_data1
df2['主题词'] = y_data1
df2['信息熵'] = z_data1
df2.to_csv('主题词与信息熵.csv',encoding='utf-8-sig',index=False)

