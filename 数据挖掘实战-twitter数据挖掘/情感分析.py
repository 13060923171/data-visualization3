import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import time
import nltk
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#设置情感模型库
sid = SentimentIntensityAnalyzer()

#读取数据
data = pd.read_excel('第二组.xlsx')

#设置停用词库
stop_words = []
with open('常用英文停用词(NLP处理英文必备)stopwords.txt','r',encoding='utf-8')as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip().replace("'",""))


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word

#去掉标点符号
def gettext(x):
    import string
    punc = string.punctuation
    for ch in punc:
        txt = str(x).replace(ch,"")
    return txt


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
    tweet = re.sub(r'#GTCartoon:', ' ', tweet)
    return tweet





def clean_text(tweet):
    processed_tweet = []
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', ' ', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', ' ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', ' ', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    # 去掉数字
    tweet = re.sub(r'\d+', ' ', tweet)
    # 标点符号
    tweet = re.sub(r'[^A-Z^a-z^0-9^]', ' ', tweet)
    # processed_tweet.append(tweet)
    words = tweet.lower().split()
    #找符合要求的词
    words = [w for w in words if w not in stop_words]

    for word in words:
        word = preprocess_word(word)
        # if is_valid_word(word):
        processed_tweet.append(word)
    #判断词的长度，如果是空词则返回空值
    if len(processed_tweet) != 0:
        return ' '.join(processed_tweet)
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

#开始调用函数
data['new_推文'] = data['推文']
#先读取文本，然后开始清洗工作，清洗步骤1
data['new_推文'] = data['new_推文'].apply(gettext)
#清洗步骤2
data['new_推文'] = data['new_推文'].apply(preprocess_word)
#清洗步骤3
data['new_推文'] = data['new_推文'].apply(clean_text)
#删掉一些空值，把存在空值的那一行全部删除
data = data.dropna(how='any')
#开始情感判断
data['scores'] = data['new_推文'].apply(lambda commentText: sid.polarity_scores(commentText))
#读取复杂度
data['compound'] = data['scores'].apply(lambda score_dict: score_dict['compound'])
#读取负面
data['Negtive'] = data['scores'].apply(lambda score_dict: score_dict['neg'])
#读取正面
data['Postive'] = data['scores'].apply(lambda score_dict: score_dict['pos'])
#读取中立
data['Neutral'] = data['scores'].apply(lambda score_dict: score_dict['neu'])
#读取复杂度
data['comp_score'] = data['scores'].apply(emotional_judgment)
#对序列重新排序
new_data = data.reset_index(drop=True)
#保存最新文档
new_data.to_csv('新_第二组.csv',encoding="utf-8-sig",sep=',',index=None)
