import pandas as pd
import nltk
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

df1 = pd.read_csv('sum_comment.csv',encoding="utf-8-sig")


sid = SentimentIntensityAnalyzer()

f = open('C-class-fenci.txt', 'w', encoding='utf-8')
for line in df1['comment']:
    tokens = nltk.word_tokenize(line)
    # 计算关键词
    all_words = tokens
    c = Counter()
    for x in all_words:
        if len(x) > 1 and x != '\r\n':
            c[x] += 1
    # Top50
    output = ""
    # print('\n词频统计结果：')
    for (k, v) in c.most_common(30):
        # print("%s:%d"%(k,v))
        output += k + " "

    f.write(output + "\n")

else:
    f.close()



sum_counts = 0
text_list = []


def emotional_judgment(x):
    neg = x['neg']
    neu = x['neu']
    pos = x['pos']
    compound = x['compound']
    if compound == 0 and neg == 0 and pos == 0 and neu == 1:
        return 'neu'
    if compound > 0:
        if pos > neg:
            return 'pos'
        else:
            return 'neg'
    elif compound < 0:
        if pos < neg:
            return 'neg'
        else:
            return 'pos'


df1['scores'] = df1['comment'].apply(lambda commentText: sid.polarity_scores(commentText))
df1['compound'] = df1['scores'].apply(lambda score_dict: score_dict['compound'])
df1['Negtive'] = df1['scores'].apply(lambda score_dict: score_dict['neg'])
df1['Postive'] = df1['scores'].apply(lambda score_dict: score_dict['pos'])
df1['Neutral'] = df1['scores'].apply(lambda score_dict: score_dict['neu'])
df1['comp_score'] = df1['scores'].apply(emotional_judgment)
df1.to_csv('emotional_comment.csv',encoding="utf-8-sig")



