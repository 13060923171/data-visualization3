import pandas as pd
import numpy as np
# 这里使用了百度开源的成熟NLP模型来预测情感倾向
# import paddlehub as hub
from IPython.display import Image
import stylecloud
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
import re
from sklearn.model_selection import train_test_split


#生成对应的词云图
def wordclound_fx(name=None):
    #读取文件
    df = pd.read_csv('./data/data.csv')
    #删除空值内容
    df = df.dropna(subset=['内容'], axis=0)
    #定位标签的内容
    df1 = df[df['标签'] == name]
    #取掉重复值
    content = df1['内容'].drop_duplicates(keep='first')
    #再删除空值
    content = content.dropna(how='any')
    #判断文本是否为中文
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True
    #进行分词处理
    def get_cut_words(content_series):
        # 读入停用词表
        stop_words = []
        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        # 分词
        word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
        return word_num_selected
    #读取上面分词效果
    text3 = get_cut_words(content_series=content)
    #开始构建词云图
    stylecloud.gen_stylecloud(text=' '.join(text3), max_words=100,
                              collocations=False,
                              #字体格式
                              font_path='simhei.ttf',
                              #图片形状
                              icon_name='fas fa-circle',
                              # icon_name='fas fa-star',
                              #文字大小
                              size=500,
                              # palette='matplotlib.Inferno_9',
                              #图片名字
                              output_name='./data/{}-词云图.png'.format(name))
    Image(filename='./data/{}-词云图.png'.format(name))

    #读取词频
    counts = {}
    for t in text3:
        counts[t] = counts.get(t, 0) + 1
    #把词频进行统计
    ls = list(counts.items())
    #按照从小到大的顺序开始排列
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    #筛选前100的词
    for key, values in ls[:100]:
        x_data.append(key)
        y_data.append(values)
    #进行保留
    df1 = pd.DataFrame()
    df1['word'] = x_data
    df1['counts'] = y_data
    df1.to_csv('./data/{}TOP100_高频词.csv'.format(name), encoding="utf-8-sig")

#文本情感判断
def snownlp():
    df = pd.read_csv('./data/data.csv')
    df = df.dropna(subset=['内容'], axis=0)
    #调用百度开源成熟的NLP框架，对文本进行打分处理
    senta = hub.Module(name="senta_bilstm")
    #删除表情包内容
    def emjio_tihuan(x):
        x1 = str(x)
        x2 = re.sub('(\[.*?\])', "", x1)
        x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
        x4 = re.sub(r'\n', '', x3)
        return x4
    #判断是否为中文
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True
    #进行正负面判断，分值小于等于0.4的，则统计为负面，大于0.4的则为正
    def score_type(x):
        x1 = float(x)
        if x1 <= 0.4:
            return '0'
        else:
            return '1'

    #进分词处理
    def get_cut_words(str1):
        # 读入停用词表
        stop_words = []

        with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                stop_words.append(line.strip())

        # 分词
        word_num = jieba.lcut(str(str1), cut_all=False)

        # 条件筛选
        word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

        word_num_selected = " ".join(word_num_selected)

        if len(word_num_selected) != 0:
            return word_num_selected
        else:
            return np.NAN

    df['内容分词'] = df['内容'].apply(emjio_tihuan)
    df = df.dropna(subset=['内容分词'], axis=0)
    df['内容分词'] = df['内容分词'].apply(get_cut_words)
    df = df.dropna(subset=['内容分词'], axis=0)
    #最后把上面做好的内容，保存为csv文件
    texts = df['内容分词'].tolist()
    input_data = {'text': texts}
    res = senta.sentiment_classify(data=input_data)
    df['内容_score'] = [x['positive_probs'] for x in res]
    df['情感_类别'] = df['内容_score'].apply(score_type)
    df.to_csv('./data/nlp_all_data.csv',encoding='utf-8-sig',index=False)


#绘制主题强度饼图
def means_pie():
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    df = pd.read_csv('./data/聚类结果.csv')
    new_df = df['聚类结果'].value_counts()
    x_data = list(new_df.index)
    y_data = list(new_df.values)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('聚类结果占比')
    plt.tight_layout()
    plt.savefig('./data/聚类结果占比.png')

# def data_class():
#     df = pd.read_csv('./data/nlp_all_data.csv')
#     data = list(df['内容分词'])
#     target = list(df['情感_类别'])
#     train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3)
#
#     with open('./BI-lstm/data/train.words.txt','w',encoding='utf-8-sig') as f:
#         for t in train_x:
#             f.write(t+'\n')
#
#     with open('./BI-lstm/data/test.words.txt','w',encoding='utf-8-sig') as f:
#         for t in test_x:
#             f.write(t+'\n')
#
#     with open('./BI-lstm/data/eval.labels.txt','w',encoding='utf-8-sig') as f:
#         for t in train_y:
#             f.write(t+'\n')
#
#     with open('./BI-lstm/data/eval.labels.txt','w',encoding='utf-8-sig') as f:
#         for t in test_y:
#             f.write(t+'\n')


if __name__ == '__main__':
    # wordclound_fx('chatgpt')
    # snownlp()
    means_pie()