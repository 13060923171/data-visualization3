import MeCab
import re
import pandas as pd
import numpy as np

# 加载停用词
def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f]
    return stopwords


# 分词并去除停用词和标点符号
def tokenize(text):
    tagger = MeCab.Tagger('-Ochasen')
    parsed = tagger.parse(text)
    words = []
    for line in parsed.splitlines()[:-1]:
        surface, feature = line.split('\t')[:2]
        if surface not in stopwords and not re.match(r'[、。．，,\.\s]+', surface) and len(surface) >= 2:
            words.append(surface)
    if len(words) != 0:
        return ' '.join(words)
    else:
        return np.NAN


if __name__ == '__main__':
    stopwords_file = 'stopwords.txt'
    stopwords = load_stopwords(stopwords_file)
    df = pd.read_excel('data.xlsx')
    df['新内容'] = df['内容'].apply(tokenize)
    new_df = df.dropna(subset=['新内容'], axis=0)
    new_df.to_excel('new_data.xlsx',index=False)