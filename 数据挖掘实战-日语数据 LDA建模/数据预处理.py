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
    words = []
    tagger = MeCab.Tagger('-Ochasen')
    parsed = tagger.parse(text).split("\n")
    for i in parsed:
        if i == 'EOS' or i == '':
            continue
        feature = i.split("\t")[3]
        surface = i.split("\t")[0]
        if "名詞" in feature or "形容" in feature or "組織名" in feature or "人名" in feature or "地名" in feature or "動詞" in feature or "副詞" in feature or "未知語" in feature or "タダ" in feature or "感動詞" in feature:
            if surface not in stopwords and not re.match(r'[、。．，,\.\s]+', surface) and len(surface) >= 2:
                words.append(surface)
    if len(words) != 0:
        return ' '.join(words)
    else:
        return np.NAN
    # for line in parsed.splitlines()[:-1]:
    #     surface = line.split('\t')[0]
    #     print(surface)
    #     if surface not in stopwords and not re.match(r'[、。．，,\.\s]+', surface) and len(surface) >= 2:
    #         words.append(surface)
    # if len(words) != 0:
    #     return ' '.join(words)
    # else:
    #     return np.NAN


if __name__ == '__main__':
    stopwords_file = 'stopwords.txt'
    stopwords = load_stopwords(stopwords_file)
    df = pd.read_excel('data.xlsx')
    df['新内容'] = df['内容'].apply(tokenize)
    new_df = df.dropna(subset=['新内容'], axis=0)
    new_df.to_excel('new_data.xlsx',index=False)