# 安装必要的库
# !pip install mecab-python3
# !pip install transformers
# !pip install pandas
import MeCab
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

def main1(text1):
    def preprocess_text(text):
        return mecab.parse(text).strip()
    # 预处理文本
    preprocessed_text = preprocess_text(text1)

    # 执行情感分析
    result = sentiment_analyzer(preprocessed_text)
    label = result[0]['label']
    score = result[0]['score']
    return label,score


if __name__ == '__main__':
    # 初始化 MeCab 进行分词
    mecab = MeCab.Tagger("-Owakati")
    # 使用 transformers 中的预训练情感分析模型
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    df = pd.read_excel('twitter_data.xlsx')
    data = pd.DataFrame()
    data['推文内容'] = ['推文内容']
    data['关键词'] = ['关键词']
    data['发布时间'] = ['发布时间']
    data['label'] =['label']
    data['score'] = ['score']
    data.to_csv('data.csv', encoding='utf-8-sig', index=False, mode='w',header=False)
    for d1,d2,d3 in tqdm(zip(df['推文内容'],df['关键词'],df['发布时间'])):
        text = d1
        label,score = main1(text)
        data = pd.DataFrame()
        data['推文内容'] = [text]
        data['关键词'] = [d2]
        data['发布时间'] = [d3]
        data['label'] = [label]
        data['score'] = [score]
        data.to_csv('data.csv',encoding='utf-8-sig',index=False,mode='a+',header=False)