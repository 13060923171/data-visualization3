import pandas as pd
import re
import jieba
import jieba.posseg as pseg
import os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 导入停用词列表
stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())

def is_all_chinese_or_english(strs):
    for _char in strs:
        # 如果字符是中文或者英文，继续检查下一个字符
        if '\u4e00' <= _char <= '\u9fa5' or '\u0041' <= _char <= '\u005a' or '\u0061' <= _char <= '\u007a':
            continue
        # 如果字符既不是中文也不是英文，返回False
        else:
            return False
    # 所有字符都是中文或者英文，返回True
    return True

def user_image1():
    if not os.path.exists("./{}".format('用户画像')):
        os.mkdir("./{}".format('用户画像'))

    df = pd.read_csv('data.csv')
    new_df = df['MBTI'].value_counts()
    for x in new_df.index:
        df1 = df[df['MBTI'] == x]
        if not os.path.exists("./{}/{}".format('用户画像',x)):
            os.mkdir("./{}/{}".format('用户画像',x))
        d = {}
        list_text = []
        for t in df1['分词']:
            # 把数据分开
            t = str(t).split(" ")
            for i in t:
                # 对文本进行分词和词性标注
                # 添加到列表里面
                list_text.append(i)
                d[i] = d.get(i, 0) + 1

        ls = list(d.items())
        ls.sort(key=lambda x: x[1], reverse=True)
        x_data = []
        y_data = []
        for key, values in ls[:100]:
            x_data.append(key)
            y_data.append(values)

        data = pd.DataFrame()
        data['word'] = x_data
        data['counts'] = y_data

        data.to_csv('./{}/{}/高频词Top100.csv'.format('用户画像',x), encoding='utf-8-sig', index=False)
        # 将词频数据转换为使用空格隔开的词汇，词频越高的词汇出现次数越多

        # 读取背景图片
        background_Image = np.array(Image.open('image.jpg'))
        # 提取背景图片颜色
        img_colors = ImageColorGenerator(background_Image)
        text = ' '.join(list_text)
        wc = WordCloud(
            collocations=False,
            font_path='simhei.ttf',
            margin=1,  # 页面边缘
            mask=background_Image,
            scale=10,
            max_words=100,  # 最多词个数
            random_state=42,
            width=900,
            height=600,
            background_color='#D4EFDF',  # 背景颜色
        )
        # 生成词云
        wc.generate_from_text(text)

        # 设置为背景色，若不想要背景图片颜色，就注释掉
        wc.recolor(color_func=img_colors)

        # 存储图像
        wc.to_file("./{}/{}/用户画像.png".format('用户画像',x))

def user_image2():
    if not os.path.exists("./{}".format('用户画像')):
        os.mkdir("./{}".format('用户画像'))

    df = pd.read_csv('data.csv')
    new_df = df['MBTI'].value_counts()
    for x in new_df.index:
        df1 = df[df['MBTI'] == x]
        if not os.path.exists("./{}/{}".format('用户画像',x)):
            os.mkdir("./{}/{}".format('用户画像',x))
        d = {}
        list_text = []
        for t in df1['评论']:
            # 对文本进行分词和词性标注
            words = pseg.cut(t)
            # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
            for word, flag in words:
                if 'a' in flag or 'A' in flag:
                    if word not in stop_words and is_all_chinese_or_english(word) == True:
                        # 如果是形容词，就将其保存到列表中
                        list_text.append(word)
                        d[word] = d.get(word, 0) + 1

        ls = list(d.items())
        ls.sort(key=lambda x: x[1], reverse=True)
        x_data = []
        y_data = []
        for key, values in ls[:30]:
            x_data.append(key)
            y_data.append(values)

        data = pd.DataFrame()
        data['word'] = x_data
        data['counts'] = y_data

        data.to_csv('./{}/{}/形容词Top30.csv'.format('用户画像',x), encoding='utf-8-sig', index=False)
        # 将词频数据转换为使用空格隔开的词汇，词频越高的词汇出现次数越多

        # 读取背景图片
        background_Image = np.array(Image.open('image.jpg'))
        # 提取背景图片颜色
        img_colors = ImageColorGenerator(background_Image)
        text = ' '.join(list_text)
        wc = WordCloud(
            collocations=False,
            font_path='simhei.ttf',
            margin=1,  # 页面边缘
            mask=background_Image,
            scale=10,
            max_words=30,  # 最多词个数
            random_state=42,
            width=900,
            height=600,
            background_color='#FAE5D3',  # 背景颜色
        )
        # 生成词云
        wc.generate_from_text(text)

        # 设置为背景色，若不想要背景图片颜色，就注释掉
        wc.recolor(color_func=img_colors)

        # 存储图像
        wc.to_file("./{}/{}/用户行为分析.png".format('用户画像',x))

def user_tfidf():
    if not os.path.exists("./{}".format('用户画像')):
        os.mkdir("./{}".format('用户画像'))

    df = pd.read_csv('data.csv')
    new_df = df['MBTI'].value_counts()
    for x in new_df.index:
        df1 = df[df['MBTI'] == x]
        if not os.path.exists("./{}/{}".format('用户画像',x)):
            os.mkdir("./{}/{}".format('用户画像',x))

        corpus = []
        for i in df1['分词']:
            corpus.append(i.strip())

            # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
        vectorizer = CountVectorizer()

        # 该类会统计每个词语的tf-idf权值
        transformer = TfidfTransformer()

        # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        # 获取词袋模型中的所有词语
        word = vectorizer.get_feature_names_out()

        # 将tf-idf矩阵抽取出来 元素w[i][j]表示j词在i类文本中的tf-idf权重
        weight = tfidf.toarray()

        data = {'word': word,
                'tfidf': weight.sum(axis=0).tolist()}

        df2 = pd.DataFrame(data)
        df2['tfidf'] = df2['tfidf'].astype('float64')
        df2 = df2.sort_values(by=['tfidf'], ascending=False)
        df2.to_csv('./{}/{}/tfidf.csv'.format('用户画像',x), encoding='utf-8-sig', index=False)

        df2 = df2.iloc[:70]

        list_word = []
        for i, j in zip(df2['word'], df2['tfidf']):
            if i not in stop_words:
                list_word.append((i, int(j)))
        # 将词频数据转换为使用空格隔开的词汇，词频越高的词汇出现次数越多
        text = ' '.join(word for word, freq in list_word for _ in range(freq))

        # 读取背景图片
        background_Image = np.array(Image.open('image.jpg'))
        # 提取背景图片颜色
        img_colors = ImageColorGenerator(background_Image)
        wc = WordCloud(
            collocations=False,
            font_path='simhei.ttf',
            margin=1,  # 页面边缘
            mask=background_Image,
            scale=10,
            max_words=70,  # 最多词个数
            random_state=42,
            width=900,
            height=600,
            background_color = '#D5D8DC', # 背景颜色

        )
        # 生成词云
        wc.generate_from_text(text)

        # 设置为背景色，若不想要背景图片颜色，就注释掉
        wc.recolor(color_func=img_colors)

        # 存储图像
        wc.to_file("./{}/{}/用户特征分析.png".format('用户画像',x))

if __name__ == '__main__':
    user_image1()
    user_image2()
    user_tfidf()