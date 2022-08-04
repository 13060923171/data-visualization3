# 中文文本分类
import os
import jieba
import warnings
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
import nltk
import re
from nltk.stem.snowball import SnowballStemmer  # 返回词语的原型，去掉ing等
from googletrans import Translator
stemmer = SnowballStemmer("english")
warnings.filterwarnings('ignore')
translator = Translator()
def cut_words(text):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces

def loadfile(file_dir, label):
    """
    将路径下的所有文件加载
    :param file_dir: 保存txt文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    """
    words_list = []
    labels_list = []
    for file in file_dir:
        words_list.append(cut_words(file))
        labels_list.append(label)
    return words_list, labels_list

stop_words = []
with open('常用英文停用词(NLP处理英文必备)stopwords.txt','r',encoding='utf-8')as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip().replace("'",""))

def tokenize_only(text):  # 分词器，仅分词
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    text_with_spaces = ''.join(filtered_tokens)
    return text_with_spaces

def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def gettext(x):
    import string
    punc = string.punctuation
    for ch in punc:
        txt = str(x).replace(ch,"")
    return txt


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
    tweet = re.sub(r'#GTCartoon:', ' ', tweet)
    return tweet


def clean_text(tweet):
    processed_tweet = []
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', ' ', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', ' ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', ' ', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    # 去掉数字
    tweet = re.sub(r'\d+', ' ', tweet)
    # 标点符号
    # tweet = re.sub(r'[^A-Z^a-z^0-9^]', ' ', tweet)
    return tweet
    # processed_tweet.append(tweet)
    # words = tweet.lower().split()
    # words = [w for w in words]
    # for word in words:
    #     word = preprocess_word(word)
    #     # if is_valid_word(word):
    #     processed_tweet.append(word)
    # if len(processed_tweet) != 0:
    #     return ' '.join(processed_tweet)
    # else:
    #     return np.NAN



def bqth(x):
    x = str(x)
    x = x.replace('游戏内讨论','资讯讨论')
    x = x.replace('观望', '观望+').replace('期待', '期待+')
    x = x.replace('品牌口碑','口碑品牌').replace('商业化','商业化+')
    x = x.replace('口碑品牌', '口碑品牌+')
    x = x.replace('HoK/AoV版本对比', '产品对比').replace('同赛道产品对比', '产品对比').replace('HoK/Aov版本对比', '产品对比')
    x = x.replace('无脑黑', '游戏故障/BUG反馈')
    return x


def bqth2(x):
    x = str(x)
    x = x.replace('期待+','期待/观望')
    x = x.replace('观望+', '期待/观望')
    x = x.replace('口碑品牌+','口碑品牌/商业化')
    x = x.replace('商业化+', '口碑品牌/商业化')
    if '期待/观望' in x or '产品对比' in x or '资讯讨论' in x or '期待/观望' in x or '游戏故障/BUG反馈' in x or '口碑品牌/商业化' in x:
        return x
    else:
        return '其他评论'


def fy(x):
    try:
        translations = translator.translate(x, dest='en')
        return translations.text
    except:
        return np.NAN

def train_data():
    df1 = pd.read_excel('数据集.xlsx',sheet_name='hok1')
    df2 = pd.read_excel('数据集.xlsx',sheet_name='hok2')
    df3 = pd.concat([df1,df2],axis=0)
    df3['一级标签'] = df3['一级标签'].apply(bqth)
    df3['一级标签'] = df3['一级标签'].apply(bqth2)
    df3['内容'] = df3['评论内容（仅保留有效内容）'].apply(fy)
    df3['内容'] = df3['内容'].apply(gettext)
    df3['内容'] = df3['内容'].apply(preprocess_word)
    df3['内容'] = df3['内容'].apply(clean_text)
    df3 = df3.dropna(how='any',axis=0)
    df3.to_csv('训练集.csv',index=None,encoding='utf-8-sig')
    new_df = df3['一级标签'].value_counts()

    x_data = list(new_df.index)

    train_words_list = []
    train_labels = []

    for x in x_data:
        data = df3['内容'][df3['一级标签'] == x]
        train_words_list1, train_labels1 = loadfile(data, x)
        train_words_list += train_words_list1
        train_labels += train_labels1
    return train_words_list,train_labels


def test_data():
    df3 = pd.read_excel('数据集.xlsx', sheet_name='hok3')
    df3['一级标签'] = df3['一级标签'].apply(bqth)
    df3['一级标签'] = df3['一级标签'].apply(bqth2)
    df3['内容'] = df3['评论内容（仅保留有效内容）'].apply(fy)
    print(df3['内容'])
    df3['内容'] = df3['内容'].apply(gettext)
    df3['内容'] = df3['内容'].apply(preprocess_word)
    df3['内容'] = df3['内容'].apply(clean_text)
    df3 = df3.dropna(how='any', axis=0)
    df3.to_csv('测试集.csv', index=None, encoding='utf-8-sig')
    new_df = df3['一级标签'].value_counts()
    x_data = list(new_df.index)

    train_words_list = []
    train_labels = []

    for x in x_data:
        data = df3['内容'][df3['一级标签'] == x]
        train_words_list1, train_labels1 = loadfile(data, x)
        train_words_list += train_words_list1
        train_labels += train_labels1
    return train_words_list, train_labels


stop_words = open('常用英文停用词(NLP处理英文必备)stopwords.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
stop_words = stop_words.split('\n') # 根据分隔符分隔

train_words_list,train_labels = train_data()
test_words_list,test_labels = test_data()

# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

train_features = tf.fit_transform(train_words_list)
# 上面fit过了，这里transform
test_features = tf.transform(test_words_list)

# 多项式贝叶斯分类器

clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
predicted_labels=clf.predict(test_features)

# result = pd.concat((df11, pd.DataFrame(predicted_labels)), axis=1)
# result.rename({0: '分类结果'}, axis=1, inplace=True)
# result.to_csv('new_class.csv',encoding="utf-8-sig")
# 计算准确率
print('分类准确率为：', metrics.accuracy_score(test_labels, predicted_labels))


