import pandas as pd
from googletrans import Translator
import random
import time
from tqdm import tqdm
import nltk
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
translator = Translator()
a = random.uniform(0.1,0.5)

df = pd.read_excel('result_网站.xlsx')

list_trans = []
for d in tqdm(df['xcl']):
    try:
        translations = translator.translate(d, dest='en')
        list_trans.append(translations.text)
        time.sleep(a)
    except:
        list_trans.append('')
        continue


def emotional_judgment(x):
    neg = x['neg']
    neu = x['neu']
    pos = x['pos']
    compound = x['compound']
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


df['英文翻译'] = list_trans
df['scores'] = df['英文翻译'].apply(lambda commentText: sid.polarity_scores(commentText))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['Negtive'] = df['scores'].apply(lambda score_dict: score_dict['neg'])
df['Postive'] = df['scores'].apply(lambda score_dict: score_dict['pos'])
df['Neutral'] = df['scores'].apply(lambda score_dict: score_dict['neu'])
df['comp_score'] = df['scores'].apply(emotional_judgment)
df.to_csv('new_data.csv',encoding="utf-8-sig",index=False)
