from docx import Document
from os import path
import re
import os
import pandas as pd
from tqdm import tqdm

from collections import Counter
import itertools

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import nltk

def main1(name):
    def read_docx(file_name):
        document = Document(file_name)
        text = ''
        for paragraph in document.paragraphs:
            text += paragraph.text
        return text

    file_name = u'C:\\Users\\v_yuhaozeng\\Desktop\\doc数据提取\\数据\\{}'.format(name)
    content = read_docx(file_name)
    df = pd.DataFrame()
    df['content'] = [content]
    df.to_csv('data.csv', encoding='utf-8-sig', mode="a+", header=False, index=False)


def main2():
    df = pd.read_csv('data.csv')
    lemmatizer = WordNetLemmatizer()
    # 设置情感模型库
    sid = SentimentIntensityAnalyzer()

    stop_words = []
    with open('常用英文停用词(NLP处理英文必备)stopwords.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip().replace("'", ""))

    # 去掉标点符号，以及机械压缩
    def preprocess_word(word):
        # Remove punctuation
        word = word.strip('\'"?!,.():;')
        # Convert more than 2 letter repetitions to 2 letter
        # funnnnny --> funny
        word = re.sub(r'(.)\1+', r'\1\1', word)
        # Remove - & '
        word = re.sub(r'(-|\')', '', word)
        return word

    # 再次去掉标点符号
    def gettext(x):
        import string
        punc = string.punctuation
        for ch in punc:
            txt = str(x).replace(ch, "")
        return txt

    #替换表情包
    def handle_emojis(text):
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' ', text)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' ', text)
        # Love -- <3, :*
        text = re.sub(r'(<3|:\*)', ' ', text)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' ', text)
        # Sad -- :-(, : (, :(, ):, )-:
        text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' ', text)
        # Cry -- :,(, :'(, :"(
        text = re.sub(r'(:,\(|:\'\(|:"\()', ' ', text)
        # Angry -- >:(, >:-(, :'(
        text = re.sub(r'(>:\(|>:-\(|:\'\()', ' ', text)
        # Surprised -- :O, :o, :-O, :-o, :0, 8-0
        text = re.sub(r'(:\s?[oO]|:-[oO]|:0|8-0)', ' ', text)
        # Confused -- :/, :\, :-/, :-\
        text = re.sub(r'(:\\|:/|:-\\\\|:-/)', ' ', text)
        # Embarrassed -- :$, :-$
        text = re.sub(r'(:\\$|:-\\$)', ' ', text)
        # Other emoticons
        emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        for emoticon in emoticons:
            text = text.replace(emoticon, " ")
        return text

    def clean_text(text):
        # Replaces URLs with the word URL
        text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', text)
        # Replace @username with the word USER_MENTION
        text = re.sub(r'@[\S]+', ' ', text)
        # Replace #hashtag with the word HASHTAG
        text = re.sub(r'#(\S+)', ' ', text)
        text = re.sub(r'#\w+', ' ', text)
        # Remove RT (retext)
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

    df = df.drop_duplicates(subset=['content'])
    df['content'] = df['content'].astype('str')
    df = df.dropna(subset=['content'], axis=0)
    df['clearn_text'] = df['content'].apply(gettext)
    df = df.dropna(subset=['clearn_text'], axis=0)
    df['clearn_text'] = df['clearn_text'].apply(clean_text)
    df.dropna(subset=['clearn_text'], axis=0, inplace=True)
    df.to_excel('data.xlsx', index=False)


if __name__ == '__main__':
    def get_all_file_names(directory_path):
        return os.listdir(directory_path)

    directory_path = "数据"  # 你的文件夹路径
    file_names = get_all_file_names(directory_path)
    df = pd.DataFrame()
    df['content'] = ['content']
    df.to_csv('data.csv', encoding='utf-8-sig', mode="w", header=False, index=False)
    for i in tqdm(file_names):
        main1(i)

    main2()

