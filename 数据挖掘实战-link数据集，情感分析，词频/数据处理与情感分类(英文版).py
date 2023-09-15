import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import time
import stylecloud
from IPython.display import Image
import os
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


def demo(name):
    df = df5[df5['search_company'] == name]
    #设置情感模型库
    sid = SentimentIntensityAnalyzer()

    lemmatizer = WordNetLemmatizer()


    stop_words = []
    with open('常用英文停用词(NLP处理英文必备)stopwords.txt','r',encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip().replace("'",""))

    #去掉标点符号，以及机械压缩
    def preprocess_word(word):
        # Remove punctuation
        word = word.strip('\'"?!,.():;')
        # Convert more than 2 letter repetitions to 2 letter
        # funnnnny --> funny
        word = re.sub(r'(.)\1+', r'\1\1', word)
        # Remove - & '
        word = re.sub(r'(-|\')', '', word)
        return word

    #再次去掉标点符号
    def gettext(x):
        import string
        punc = string.punctuation
        for ch in punc:
            txt = str(x).replace(ch,"")
        return txt


    #替换表情包
    def handle_emojis(tweet):
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' ', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' ', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', ' ', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' ', tweet)
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' ', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' ', tweet)
        # Angry -- >:(, >:-(, :'(
        tweet = re.sub(r'(>:\(|>:-\(|:\'\()', ' ', tweet)
        # Surprised -- :O, :o, :-O, :-o, :0, 8-0
        tweet = re.sub(r'(:\s?[oO]|:-[oO]|:0|8-0)', ' ', tweet)
        # Confused -- :/, :\, :-/, :-\
        tweet = re.sub(r'(:\\|:/|:-\\\\|:-/)', ' ', tweet)
        # Embarrassed -- :$, :-$
        tweet = re.sub(r'(:\\$|:-\\$)', ' ', tweet)
        # Other emoticons
        # This regex matching method is taken from Twitter NLP utils:
        # https://github.com/aritter/twitter_nlp/blob/master/python/emoticons.py
        emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', tweet)
        for emoticon in emoticons:
            tweet = tweet.replace(emoticon, " ")
        return tweet


    def clean_text(text):
        # Replaces URLs with the word URL
        url_pattern = re.compile(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+')
        text = url_pattern.sub('', text)
        # Replace @username with the word USER_MENTION
        text = re.sub(r'@[\S]+', ' ', text)
        # Replace #hashtag with the word HASHTAG
        text = re.sub(r'#(\S+)', ' ', text)
        # Remove RT (retweet)
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

    # 定义机械压缩函数
    def yasuo(st):
        for i in range(1, int(len(st) / 2) + 1):
            for j in range(len(st)):
                if st[j:j + i] == st[j + i:j + 2 * i]:
                    k = j + i
                    while st[k:k + i] == st[k + i:k + 2 * i] and k < len(st):
                        k = k + i
                    st = st[:j] + st[k:]
        return st

    #情感判断
    def emotional_judgment(x):
        compound = float(x)
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'


    df['commentary'] = df['commentary'].apply(gettext)
    df['commentary'] = df['commentary'].apply(yasuo)
    df['clearn_comment'] = df['commentary'].apply(preprocess_word)
    df['clearn_comment'] = df['clearn_comment'].apply(clean_text)
    df.dropna(subset=['clearn_comment'],axis=0,inplace=True)
    #开始情感判断
    df['sentiment'] = df['clearn_comment'].apply(lambda commentText: sid.polarity_scores(commentText))
    # 提取复合分数
    df['compound'] = df['sentiment'].apply(lambda score_dict: score_dict['compound'])
    #读取复杂度
    df['comp_score'] = df['compound'].apply(emotional_judgment)

    if not os.path.exists("./{}".format(name)):
            os.mkdir("./{}".format(name))
    new_data = df['comp_score'].value_counts()
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    x_data = list(new_data.index)
    y_data = list(new_data.values)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('emotion type')
    plt.tight_layout()
    plt.savefig('./{}/情感分布情况.png'.format(name))



    d = {}
    list_text = []
    for t in df['clearn_comment']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            # 再过滤一遍无效词
            if i not in stop_words:
                # 添加到列表里面
                list_text.append(i)
                d[i] = d.get(i,0)+1

    ls = list(d.items())
    ls.sort(key=lambda x:x[1],reverse=True)
    x_data = []
    y_data = []
    for key,values in ls[:100]:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data
    data.to_csv('./{}/评论高频词Top100.csv'.format(name),encoding='utf-8-sig',index=False)
    # 然后传入词云图中，筛选最多的100个词
    stylecloud.gen_stylecloud(text=' '.join(list_text), max_words=100,
                              # 不能有重复词
                              collocations=False,
                              max_font_size=400,
                              # 字体样式
                              font_path='simhei.ttf',
                              # 图片形状
                              icon_name='fas fa-crown',
                              # 图片大小
                              size=1200,
                              # palette='matplotlib.Inferno_9',
                              # 输出图片的名称和位置
                              output_name='./{}/词云图.png'.format(name))
    # 开始生成图片
    Image(filename='./{}/词云图.png'.format(name))

    df['length'] = df['commentary'].apply(lambda x:len(x))
    x_data = ['average','median']
    y_data = [round(df['length'].mean(),2),round(df['length'].median(),2)]
    plt.figure(figsize=(9,6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.bar(x_data,y_data)
    for i, value in enumerate(y_data):
        plt.annotate(value, # 数值
                     (x_data[i], value), # 放置数值的坐标
                     textcoords="offset points", # 文本坐标系
                     xytext=(0,5), # 文本偏移量
                     ha='center') # 横向对齐方式

    plt.title("Statement length statistics")
    plt.ylabel("length")# 显示数值

    plt.savefig('./{}/Statement length statistics.png'.format(name))

    df.to_excel('./{}/new_data.xlsx'.format(name))


if __name__ == '__main__':
    df5 = pd.read_excel('company_post.xlsx')
    new_df = df5['search_company'].value_counts()
    names = [x for x in new_df.index]
    for name in names:
        demo(name)