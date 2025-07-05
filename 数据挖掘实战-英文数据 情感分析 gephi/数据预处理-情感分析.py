import re
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk

from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import torch
from tqdm import tqdm


# 加载NLTK的停用词列表
nltk_stop_words = set(stopwords.words('english'))

stop_words = []
with open('stopwords_en.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip().replace("'", ""))
stop_words = set(stop_words)


# 去掉标点符号，以及机械压缩
def preprocess_word(word):
    word = word.strip('\'"?!,.():;-')
    word = re.sub(r'(.)\1+', r'\1\1', word)
    word = re.sub(r'(-|\')', '', word)
    return word


# 替换表情包
def handle_emojis(text):
    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' ', text)
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' ', text)
    text = re.sub(r'(<3|:\*)', ' ', text)
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' ', text)
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' ', text)
    text = re.sub(r'(:,\(|:\'\(|:"\()', ' ', text)
    text = re.sub(r'(>:\(|>:-\(|:\'\()', ' ', text)
    text = re.sub(r'(:\s?[oO]|:-[oO]|:0|8-0)', ' ', text)
    text = re.sub(r'(:\\|:/|:-\\\\|:-/)', ' ', text)
    text = re.sub(r'(:\\$|:-\\$)', ' ', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    for emoticon in emoticons:
        text = text.replace(emoticon, " ")
    return text


# 处理文本
def clean_text(text):
    text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', text)
    text = re.sub(r'@[\S]+', ' ', text)
    text = re.sub(r'#(\S+)', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'\brt\b', ' ', text)
    text = re.sub(r'\.{2,}', ' ', text)
    text = text.strip(' "\'')
    text = handle_emojis(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
    text = text.lower()

    words = text.split()
    words = [w for w in words if w not in stop_words and w not in nltk_stop_words]
    words = [preprocess_word(w) for w in words]

    # 保留名词和形容词
    pos_tags = pos_tag(words)
    filtered_words = [word for word, pos in pos_tags if len(word) >= 3 and (pos.startswith('NN') or pos.startswith('JJ'))]

    #词性还原
    lemmatizer = WordNetLemmatizer()

    # 应用lemmatizer进行词形还原
    line = [lemmatizer.lemmatize(word) for word in filtered_words]

    if len(line) != 0:
        return ' '.join(line)
    else:
        return np.NAN


data = pd.read_excel('post.xlsx')
df = data[data['语言类型'] == 'en']
df = df.drop_duplicates(subset=['发布内容'])
df['发布内容'] = df['发布内容'].astype('str')
df = df.dropna(subset=['发布内容'], axis=0)
df['分词'] = df['发布内容'].apply(clean_text)
df['分词'] = df['分词'].replace('nan', np.NAN)
df.dropna(subset=['分词'], axis=0, inplace=True)


device = 0 if torch.cuda.is_available() else -1  # 使用GPU（如果可用）
# 使用更高效的模型和GPU
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)

# 并行化处理文本
def classify_text(text):
    return classifier(text)

texts = df['分词'].tolist()
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(classify_text, texts))

list_label = []
list_score = []
for result in tqdm(results):
    label = result[0]['label']
    score = result[0]['score']
    list_label.append(label)
    list_score.append(score)

# 添加结果到DataFrame
df['情感类别'] = list_label
df['情感得分'] = list_score

df.to_excel('new_post.xlsx', index=False)



