import re

import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
# 数据处理库
import pyLDAvis
import pyLDAvis.gensim
import stylecloud
from IPython.display import Image
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#设置情感模型库
sid = SentimentIntensityAnalyzer()
#在这里修改文件名
df = pd.read_excel('媒体数据.xlsx')
lemmatizer = WordNetLemmatizer()
stop_words = ['ne','zha','br','ao','de','la','zhas','yuan','ult','don','el','gt','toy','uk','li','jian','yu']
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

#情感判断
def emotional_judgment(x):
    #读取负面的值
    neg = x['neg']
    #读取中立的值
    neu = x['neu']
    #读取正面的值
    pos = x['pos']
    #读取复杂度
    compound = x['compound']
    #判断复杂度的表现情况，从而判断是属于中立还是正面还是负面
    if compound == 0 and neg == 0 and pos == 0 and neu == 1:
        return 'neu'
    elif compound > 0:
        if pos > neg:
            return 'pos'
        else:
            return 'neg'
    elif compound < 0:
        if pos < neg:
            return 'neg'
        else:
            return 'pos'
    else:
        return 'neu'

df['清洗文本'] = df['文本内容'].apply(gettext)
df['清洗文本'] = df['清洗文本'].apply(preprocess_word)
df['清洗文本'] = df['清洗文本'].apply(clean_text)
df.dropna(subset=['清洗文本'],axis=0,inplace=True)
#开始情感判断
df['scores'] = df['清洗文本'].apply(lambda commentText: sid.polarity_scores(commentText))
#读取复杂度
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
#读取负面
df['Negtive'] = df['scores'].apply(lambda score_dict: score_dict['neg'])
#读取正面
df['Postive'] = df['scores'].apply(lambda score_dict: score_dict['pos'])
#读取中立
df['Neutral'] = df['scores'].apply(lambda score_dict: score_dict['neu'])
#读取复杂度
df['comp_score'] = df['scores'].apply(emotional_judgment)
#保存最新文档
df.to_excel('新_媒体数据.xlsx',encoding="utf-8-sig",index=None)
d = {}
list_text = []
for t in df['清洗文本']:
    # 把数据分开
    t = str(t).split(" ")
    for i in t:
        # 再过滤一遍无效词
        if i not in stop_words:
            # 添加到列表里面
            list_text.append(i)
            d[i] = d.get(i,0)+1

ls = list(d.items())
ls.sort(key=lambda x:x[1],reverse=True)
x_data = []
y_data = []
for key,values in ls[:200]:
    x_data.append(key)
    y_data.append(values)

data = pd.DataFrame()
data['word'] = x_data
data['counts'] = y_data
data.to_csv('评论高频词Top200.csv',encoding='utf-8-sig',index=False)
# 然后传入词云图中，筛选最多的100个词
stylecloud.gen_stylecloud(text=' '.join(list_text), max_words=100,
                          # 不能有重复词
                          collocations=False,
                          max_font_size=400,
                          # 字体样式
                          font_path='simhei.ttf',
                          # 图片形状
                          icon_name='fas fa-crown',
                          # 图片大小
                          size=1200,
                          # palette='matplotlib.Inferno_9',
                          # 输出图片的名称和位置
                          output_name='媒体数据-词云图.png')
# 开始生成图片
Image(filename='媒体数据-词云图.png')




new_data = df['comp_score'].value_counts()
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(dpi=500)
x_data = list(new_data.index)
y_data = list(new_data.values)
plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
plt.title('emotion type')
plt.tight_layout()
plt.savefig('媒体数据-情感分布情况.png')