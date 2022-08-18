import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from imageio import imread
import nltk
import re
from nltk.stem.snowball import SnowballStemmer  # 返回词语的原型，去掉ing等

#加入英文词库，其目的是为了把部分单词还原
stemmer = SnowballStemmer("english")
stop_words = []
#加入停用词库
with open("常用英文停用词(NLP处理英文必备)stopwords.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())
#可以对停用词进行新增，其目的是去除自己不需要的词
new_stop_words = []
stop_words.extend(new_stop_words)


def tokenize_only(text):  # 分词器，仅分词
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def main1():
    # 绘制词云图
    df1 = pd.read_csv('新_第二组.csv')
    sum_cotent = []
    #读取文本数据，并且对文本进行分词处理
    df1['new_推文'] = df1['new_推文'].apply(tokenize_only)
    df1['new_推文'].astype('str')
    y_data1 = list(df1['new_推文'])
    for d in y_data1:
        for i in d:
            if i not in stop_words:
                sum_cotent.append(i)

    #对词进行统计，其目的是为了找出词频最高的前200个词
    counts = {}
    for s in sum_cotent:
        counts[s] = counts.get(s,0)+1
    ls = list(counts.items())
    ls.sort(key=lambda x:x[1],reverse=True)
    x_data = []
    y_data = []
    for key,values in ls:
        x_data.append(key)
        y_data.append(values)
    #对统计好的数据进行保存为CSV文档
    df2 = pd.DataFrame()
    df2['word'] = x_data
    df2['counts'] = y_data
    df2.to_csv('第二组_高频词top200.csv',encoding='utf-8-sig')
    contents_list = " ".join(sum_cotent)

    # 制作词云图，collocations避免词云图中词的重复，mask定义词云图的形状，图片要有背景色
    wc = WordCloud(
        collocations=False,
        max_words=100,
        background_color="white",
        font_path=r"C:\Windows\Fonts\simhei.ttf",
        stopwords=stop_words,
        width=1080, height=1920, random_state=42,
        mask=imread('1.jpg', pilmode="RGB"))
    wc.generate(contents_list)
    # 要读取的形状的图片
    wc.to_file("第二组_词云.jpg")

def main2(str1):
    # df3 = pd.read_csv('新_第一组.csv')
    df3 = pd.read_csv('./data/新_第一组.csv')
    # df3 = pd.concat([df1,df2],axis=0)
    df4 = df3[df3['comp_score'] == str1]
    sum_cotent = []
    df4['new_推文'] = df4['new_推文'].apply(tokenize_only)
    df4['new_推文'].astype('str')
    y_data1 = list(df4['new_推文'])
    for d in y_data1:
        for i in d:
            if i not in stop_words:
                sum_cotent.append(i)

    # 对词进行统计，其目的是为了找出词频最高的前200个词
    counts = {}
    for s in sum_cotent:
        counts[s] = counts.get(s, 0) + 1
    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    for key, values in ls:
        x_data.append(key)
        y_data.append(values)
    # 对统计好的数据进行保存为CSV文档
    df5 = pd.DataFrame()
    df5['word'] = x_data
    df5['counts'] = y_data
    df5.to_csv('./data/第一组_{}.csv'.format(str1), encoding='utf-8-sig')

if __name__ == '__main__':
    # main1()
    main2('pos')
