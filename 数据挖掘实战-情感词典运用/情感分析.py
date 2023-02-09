import pandas as pd


#基于波森情感词典计算情感值
def getscore(text):
    df = pd.read_table(r"情感词典.txt", sep="\t", names=['key', 'score'])
    key = df['key'].values.tolist()
    score = df['score'].values.tolist()
    # jieba分词
    segs = str(text).split(" ")
    # 计算得分
    score_list = [score[key.index(x)] for x in segs if(x in key)]
    return sum(score_list)


def main(x):
    x1 = getscore(x)
    sentiments = round(x1, 4)
    sentiments = round(sentiments / 100,4)
    return sentiments


def main1(x):
    sentiments = float(x)
    if sentiments >= 0.55:
        return "positive"
    elif 0.45 <= sentiments < 0.55:
        return "neutral"
    else:
        return "negative"


data = pd.read_csv('sum_data.csv')
data['sentiments_score'] = data['participle_process'].apply(main)
data['sentiments_decide'] = data['sentiments_score'].apply(main1)
data.to_csv('new_sum_data.csv',encoding='utf-8-sig',index=False)