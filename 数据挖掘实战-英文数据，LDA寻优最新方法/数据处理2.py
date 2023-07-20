import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import time
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import json
import ujson
import csv
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
# def json_to_csv(json_file, csv_file):
#     with open(json_file, 'r',encoding='utf-8-sig') as file:
#         data = ujson.load(file)
#
#     field_names = list(data[0].keys())
#
#     with open(csv_file, 'w', newline='',encoding='utf-8-sig') as file:
#         writer = csv.DictWriter(file, fieldnames=field_names)
#         writer.writeheader()
#         for row in data:
#             filtered_row = {key: value for key, value in row.items() if key in field_names}
#             writer.writerow(filtered_row)
#
# # 示例用法
# json_to_csv('data.json', 'data1.csv')

df = pd.read_csv('data1.csv')
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
    # This regex matching method is taken from Twitter NLP utils:
    # https://github.com/aritter/twitter_nlp/blob/master/python/emoticons.py
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


df['clearn_comment'] = df['text'].apply(gettext)
df['clearn_comment'] = df['clearn_comment'].apply(preprocess_word)
df['clearn_comment'] = df['clearn_comment'].apply(clean_text)
new_df = df.dropna(subset=['clearn_comment'],axis=0)

classifier = pipeline('sentiment-analysis')
label_list = []
score_list = []
for d in new_df['text']:
    class1 = classifier(d)
    label = class1[0]['label']
    score = class1[0]['score']
    if score <= 0.55:
        label = 'NEUTRAL'
        label_list.append(label)
    else:
        label = label
        label_list.append(label)
    score_list.append(score)

new_df['情感类型'] = label_list
new_df['情感得分'] = score_list

new_df.to_csv('new_data1.csv', encoding='utf-8-sig', index=False)

