import pandas as pd
from googletrans import Translator
import re
import numpy as np
from tqdm import tqdm
import time
from nltk.stem import WordNetLemmatizer
import matplotlib
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.models import ldamodel
from multiprocessing import freeze_support
import pyLDAvis.gensim
import pyLDAvis
from tqdm import tqdm
import warnings

# df1 = pd.read_csv('data.csv')
# df2 = pd.read_csv('data1.csv')
# df = pd.concat([df1,df2],axis=0)


# def translate_chinese_to_english(text):
#     try:
#         translator = Translator(service_urls=['translate.google.com.hk'])
#         translation = translator.translate(text, src='auto', dest='en')
#         return translation.text
#     except:
#         return text
#
#
# lemmatizer = WordNetLemmatizer()
#
#
# stop_words = []
# with open('常用英文停用词(NLP处理英文必备)stopwords.txt','r',encoding='utf-8')as f:
#     lines = f.readlines()
#     for line in lines:
#         stop_words.append(line.strip().replace("'",""))
#
# #去掉标点符号，以及机械压缩
# def preprocess_word(word):
#     # Remove punctuation
#     word = word.strip('\'"?!,.():;')
#     # Convert more than 2 letter repetitions to 2 letter
#     # funnnnny --> funny
#     word = re.sub(r'(.)\1+', r'\1\1', word)
#     # Remove - & '
#     word = re.sub(r'(-|\')', '', word)
#     return word
#
# #再次去掉标点符号
# def gettext(x):
#     import string
#     punc = string.punctuation
#     for ch in punc:
#         txt = str(x).replace(ch,"")
#     return txt
#
#
# #替换表情包
# def handle_emojis(tweet):
#     # Smile -- :), : ), :-), (:, ( :, (-:, :')
#     tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' ', tweet)
#     # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
#     tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' ', tweet)
#     # Love -- <3, :*
#     tweet = re.sub(r'(<3|:\*)', ' ', tweet)
#     # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
#     tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' ', tweet)
#     # Sad -- :-(, : (, :(, ):, )-:
#     tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' ', tweet)
#     # Cry -- :,(, :'(, :"(
#     tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' ', tweet)
#     # Angry -- >:(, >:-(, :'(
#     tweet = re.sub(r'(>:\(|>:-\(|:\'\()', ' ', tweet)
#     # Surprised -- :O, :o, :-O, :-o, :0, 8-0
#     tweet = re.sub(r'(:\s?[oO]|:-[oO]|:0|8-0)', ' ', tweet)
#     # Confused -- :/, :\, :-/, :-\
#     tweet = re.sub(r'(:\\|:/|:-\\\\|:-/)', ' ', tweet)
#     # Embarrassed -- :$, :-$
#     tweet = re.sub(r'(:\\$|:-\\$)', ' ', tweet)
#     # Other emoticons
#     # This regex matching method is taken from Twitter NLP utils:
#     # https://github.com/aritter/twitter_nlp/blob/master/python/emoticons.py
#     emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', tweet)
#     for emoticon in emoticons:
#         tweet = tweet.replace(emoticon, " ")
#     return tweet
#
#
# def clean_text(text):
#     # Replaces URLs with the word URL
#     text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', text)
#     # Replace @username with the word USER_MENTION
#     text = re.sub(r'@[\S]+', ' ', text)
#     # Replace #hashtag with the word HASHTAG
#     text = re.sub(r'#(\S+)', ' ', text)
#     # Remove RT (retweet)
#     text = re.sub(r'\brt\b', ' ', text)
#     # Replace 2+ dots with space
#     text = re.sub(r'\.{2,}', ' ', text)
#     # Strip space, " and ' from text
#     text = text.strip(' "\'')
#     # Handle emojis
#     text = handle_emojis(text)
#     # Replace multiple spaces with a single space
#     text = re.sub(r'\s+', ' ', text)
#     # Remove numbers
#     text = re.sub(r'\d+', ' ', text)
#     # Remove punctuation
#     text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
#     # Lowercase and split into words
#     words = text.lower().split()
#     words = [w for w in words if w not in stop_words]
#     words = [lemmatizer.lemmatize(w) for w in words]
#     words = [preprocess_word(w) for w in words]
#     if len(words) != 0:
#         return ' '.join(words)
#     else:
#         return np.NAN
#
# df.dropna(subset=['content'],axis=0,inplace=True)
# df['content'] = df['content'].apply(translate_chinese_to_english)
# df['clearn_comment'] = df['content'].apply(gettext)
# df['clearn_comment'] = df['clearn_comment'].apply(preprocess_word)
# df['clearn_comment'] = df['clearn_comment'].apply(clean_text)
# new_df = df.dropna(subset=['clearn_comment'],axis=0)
# new_df.to_csv('new_data.csv', encoding='utf-8-sig', index=False)


def lda():
    # 构建词典
    df = pd.read_csv('new_data.csv')
    corpus = []
    # 读取预料 一行预料为一个文档
    for d in df['clearn_comment']:
        d = str(d).split(" ")
        corpus.append(d)

    dictionary = Dictionary(corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]

    # # 定义评估指标函数
    # def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    #     coherence_values = []
    #     model_list = []
    #     # 使用tqdm显示进度条
    #     for num_topics in tqdm(range(start, limit, step)):
    #         model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
    #         model_list.append(model)
    #         #Coherence（一致性）是用来评估主题模型的一种指标，c_v方法使用了类似点互信息的计算，可以有效地衡量主题的连贯性
    #         coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    #         coherence_values.append(coherence_model.get_coherence())
    #
    #     return model_list, coherence_values
    #
    # # 调用评估函数来计算不同主题数下的模型评估指标
    # start = 2
    # limit = 16
    # step = 1
    # model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus_bow, texts=corpus,start=start, limit=limit, step=step)
    #
    # # 绘制评估指标随主题数变化的曲线
    # x = range(start, limit, step)
    # plt.plot(x, coherence_values)
    # plt.title('{}_coherence_values'.format(time_name))
    # plt.xlabel("Number of Topics")
    # plt.ylabel("Coherence Score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.savefig('coherence_values.png')
    # plt.show()
    # df = pd.DataFrame()
    # df['Topic number'] = x
    # df['coherence_values'] = coherence_values
    # df.to_csv('coherence_values.csv',encoding='utf-8-sig')
    # # 根据coherence值选择最优主题数
    # optimal_index = np.argmax(coherence_values)
    # optimal_model = model_list[optimal_index]
    # optimal_num_topics = start + optimal_index * step
    # print("Optimal number of topics:", optimal_num_topics)

    optimal_num_topics = 6
    # LDA可视化模块
    # 构建lda主题参数
    lda = ldamodel.LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=optimal_num_topics)
    # 读取lda对应的数据
    data1 = pyLDAvis.gensim.prepare(lda, corpus_bow, dictionary)
    # 把数据进行可视化处理
    pyLDAvis.save_html(data1, 'lda.html')

    # 主题判断模块
    list3 = []
    list2 = []
    # 这里进行lda主题判断
    for i in lda.get_document_topics(corpus_bow)[:]:
        listj = []
        list1 = []
        for j in i:
            list1.append(j)
            listj.append(j[1])
        list3.append(list1)
        bz = listj.index(max(listj))
        list2.append(i[bz][0])

    data = pd.DataFrame()
    data['主题概率'] = list3
    data['主题类型'] = list2

    #获取对应主题出现的频次
    new_data = data['主题类型'].value_counts()
    new_data = new_data.sort_index(ascending=True)
    y_data1 = [y for y in new_data.values]

    #主题词模块
    word = lda.print_topics(num_words=20)
    topic = []
    quanzhong = []
    list_gailv = []
    list_gailv1 = []
    list_word = []
    #根据其对应的词，来获取其相应的权重
    for w in word:
        ci = str(w[1])
        c1 = re.compile('\*"(.*?)"')
        c2 = c1.findall(ci)
        list_word.append(c2)
        c3 = '、'.join(c2)

        c4 = re.compile(".*?(\d+).*?")
        c5 = c4.findall(ci)
        for c in c5[::1]:
            if c != "0":
                gailv = str(0) + '.' + str(c)
                list_gailv.append(gailv)
        list_gailv1.append(list_gailv)
        list_gailv = []
        zt = "Topic" + str(w[0])
        topic.append(zt)
        quanzhong.append(c3)

    #把上面权重的词计算好之后，进行保存为csv文件
    df2 = pd.DataFrame()
    for j,k,l in zip(topic,list_gailv1,list_word):
        df2['{}-主题词'.format(j)] = l
        df2['{}-权重'.format(j)] = k
    df2.to_csv('主题词分布表.csv', encoding='utf-8-sig', index=False)

    y_data2 = []
    for y in y_data1:
        number = float(y / sum(y_data1))
        y_data2.append(float('{:0.5}'.format(number)))

    df1 = pd.DataFrame()
    df1['所属主题'] = topic
    df1['文章数量'] = y_data1
    df1['特征词'] = quanzhong
    df1['主题强度'] = y_data2
    df1.to_csv('特征词.csv',encoding='utf-8-sig',index=False)


#绘制主题强度饼图
def plt_pie():
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    df = pd.read_csv('特征词.csv')
    x_data = list(df['所属主题'])
    y_data = list(df['文章数量'])
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('theme strength')
    plt.tight_layout()
    plt.savefig('theme strength.png')


if __name__ == '__main__':
    lda()
    plt_pie()

