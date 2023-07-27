import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
# 数据处理库
import pyLDAvis
import pyLDAvis.gensim
import gensim
import gensim.corpora as corpora
import math
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
#nltk.download('wordnet')
#在这里修改文件名
df = pd.read_excel('result2-1689328579.xlsx')
lemmatizer = WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()

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
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', tweet)
    for emoticon in emoticons:
        tweet = tweet.replace(emoticon, " ")
    return tweet


def clean_text(text):
    # Replaces URLs with the word URL
    text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', text)
    # Replace @username with the word USER_MENTION
    text = re.sub(r'@[\S]+', ' ', text)
    # Replace #hashtag with the word HASHTAG
    text = re.sub(r'#(\S+)', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
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



#df['全文']这个指的是要读取的那一列的列名，也就是文本数据那一行
#如果Excel表格列名有改动，记得修改
df['clearn_comment'] = df['全文'].apply(gettext)
df['clearn_comment'] = df['clearn_comment'].apply(preprocess_word)
df['clearn_comment'] = df['clearn_comment'].apply(clean_text)
df.dropna(subset=['clearn_comment'],axis=0,inplace=True)
df['scores'] = df['clearn_comment'].apply(lambda commentText: sid.polarity_scores(commentText))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['Negtive'] = df['scores'].apply(lambda score_dict: score_dict['neg'])
df['Postive'] = df['scores'].apply(lambda score_dict: score_dict['pos'])
df['Neutral'] = df['scores'].apply(lambda score_dict: score_dict['neu'])
df['comp_score'] = df['scores'].apply(emotional_judgment)


dictionary = corpora.Dictionary([text.split() for text in df['clearn_comment']])
corpus = [dictionary.doc2bow(text.split()) for text in df['clearn_comment']]

#这里是主题数的输入，如果感觉主题数不满意，可以在这里重新运行，输入你需要的主题数
num_topics = input('请输入主题数:')

#LDA可视化模块
#构建lda主题参数
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111, iterations=400)
#读取lda对应的数据
data1 = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
#把数据进行可视化处理
pyLDAvis.save_html(data1, './result2/lda.html')

# 计算主题的信息熵
topic_entropies = []
for topic in lda.show_topics():
    topic_words = [word for word, prob in lda.show_topic(topic[0])]
    word_probs = [prob for word, prob in lda.show_topic(topic[0])]
    entropy = np.sum(-np.array(word_probs) * np.log2(word_probs))
    topic_entropies.append((topic[0], topic_words, entropy))

z_data1 = []
x_data1 = []
y_data1 = []
# 打印主题及其对应的信息熵
for topic_entropy in topic_entropies:
    print("Topic {}: {}".format(topic_entropy[0], topic_entropy[1]))
    x_data1.append(topic_entropy[0])
    y_data1.append(topic_entropy[1])
    print("Entropy: {}".format(topic_entropy[2]))
    z_data1.append(topic_entropy[2])

#将上面获取的数据进行保存
df2 = pd.DataFrame()
df2['主题数'] = x_data1
df2['主题词'] = y_data1
df2['信息熵'] = z_data1
#这里是文件保存路径，到时候运行不同文件，最好修改一下相对的路径，不然就是会覆盖
df2.to_csv('./result2/主题词与信息熵.csv',encoding='utf-8-sig',index=False)


def entropy(prob_list):
    # 初始化信息熵为0
    ent = 0
    # 遍历列表中的每个概率
    for p in prob_list:
        # 如果概率为0，跳过该项
        if p == 0:
            continue
        # 否则，用公式累加信息熵
        else:
            ent -= p * math.log2(p)
    # 返回信息熵
    return ent

def main1(text):
    # 对文本进行分词，用空格分隔
    words = text.split()

    # 统计每个词的出现次数，用字典存储
    word_count = {}
    for word in words:
        # 如果词已经在字典中，增加其计数
        if word in word_count:
            word_count[word] += 1
        # 否则，初始化其计数为1
        else:
            word_count[word] = 1

    # 计算每个词的概率，用列表存储
    prob_list = []
    # 获取文本的总词数
    total_words = len(words)
    # 遍历字典中的每个词和计数
    for word, count in word_count.items():
        # 计算词的概率，即计数除以总词数
        prob = count / total_words
        # 将概率添加到列表中
        prob_list.append(prob)

    # 调用之前定义的函数，计算信息熵
    ent = entropy(prob_list)
    return ent


df['entropy_values'] = df['clearn_comment'].apply(main1)


# 主题判断模块
list3 = []
list2 = []
# 这里进行lda主题判断
for i in lda.get_document_topics(corpus)[:]:
    listj = []
    list1 = []
    for j in i:
        list1.append(j)
        listj.append(j[1])
    list3.append(list1)
    bz = listj.index(max(listj))
    list2.append(i[bz][0])


df['主题概率'] = list3
df['主题类型'] = list2
#这里是文件保存路径，到时候运行不同文件，最好修改一下相对的路径，不然就是会覆盖
df.to_csv('./result2/new_data.csv', encoding='utf-8-sig', index=False)

