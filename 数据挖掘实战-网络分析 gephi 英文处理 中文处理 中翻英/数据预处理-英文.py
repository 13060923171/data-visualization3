import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

# 确保需要的NLTK资源已经下载
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

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
    word = word.strip('\'"?!,.():;')
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

    if len(filtered_words) != 0:
        return ' '.join(filtered_words)
    else:
        return np.NaN

lemmatizer = WordNetLemmatizer()

# 读取输入文件
input_file = 'new_data_trans.csv'
output_file = 'en_comment.xlsx'

df = pd.read_csv(input_file, encoding='utf-8-sig')
df['分词-英文'] = df['分词-英文'].astype('str')
df = df.dropna(subset=['分词-英文'], axis=0)
df['分词1'] = df['分词-英文'].apply(clean_text)
df['分词1'] = df['分词1'].replace('nan', np.NaN)
df.dropna(subset=['分词1'], axis=0, inplace=True)
df.to_excel(output_file, index=False)