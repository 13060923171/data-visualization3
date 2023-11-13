import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
from snownlp import SnowNLP
import threading
import concurrent.futures
from tqdm import tqdm
import os


def main(name):
    def demo1(name):
        try:
            df1 = pd.read_excel('./豆瓣/评论下载.xlsx')
            df2 = df1[df1['电影名'] == name]
            df2 = df2.dropna(subset=['评论内容'], axis=0)
            return df2['评论内容']
        except:
            df = pd.DataFrame()
            return df



    def demo2(name):
        try:
            df1 = pd.read_excel('./猫眼/评论下载.xlsx')
            df2 = df1[df1['电影名'] == name]
            df2 = df2.dropna(subset=['评论内容'], axis=0)
            return df2['评论内容']
        except:
            df = pd.DataFrame()
            return df

    def demo3(name):
        try:
            df1 = pd.read_excel('./时光网/评论下载.xlsx')
            df2 = df1[df1['电影名'] == name]
            df2 = df2.dropna(subset=['评论内容'], axis=0)
            return df2['评论内容']
        except:
            df = pd.DataFrame()
            return df

    def demo4(name):
        try:
            df1 = pd.read_excel('./微博/评论下载.xlsx')
            df2 = df1[df1['电影'] == name]
            df2 = df2.dropna(subset=['content_'], axis=0)
            return df2['content_']
        except:
            df = pd.DataFrame()
            return df

    def demo5(name):
        try:
            df1 = pd.read_excel('./知乎/帖子.xlsx')
            df2 = df1[df1['电影名称'] == name]
            df2 = df2.dropna(subset=['回答内容'],axis=0)
            return df2['回答内容']
        except:
            df = pd.DataFrame()
            return df

    def demo6(name):
        try:
            df1 = pd.read_excel('./知乎/帖子评论.xlsx')
            df2 = df1[df1['电影名称'] == name]
            df2 = df2.dropna(subset=['评论内容'], axis=0)
            return df2['评论内容']
        except:
            df = pd.DataFrame()
            return df

    data1 = demo1(name)
    data2 = demo2(name)
    data3 = demo3(name)
    data4 = demo4(name)
    data5 = demo5(name)
    data6 = demo6(name)

    data = pd.concat([data1,data2,data3,data4,data5,data6],axis=0)
    # 去除重复行
    data = data.drop_duplicates()

    df = pd.DataFrame()
    df['评论内容'] = data.values


    # 导入停用词列表
    stop_words = []
    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())


    # 判断是否为中文
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True


    def get_cut_words(content_series):
        # 对文本进行分词和词性标注
        try:
            words = pseg.cut(content_series)
            # 保存名词和形容词的列表
            nouns_and_adjs = []
            # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
            for word, flag in words:
                    if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                        # 如果是名词或形容词，就将其保存到列表中
                        nouns_and_adjs.append(word)
            if len(nouns_and_adjs) != 0:
                return ' '.join(nouns_and_adjs)
            else:
                return np.NAN
        except:
            return np.NAN

    def snownlp_content(text):
        # 初始化SnowNLP对象
        try:
            s1 = SnowNLP(text)
            number = s1.sentiments
            number = float(number)
            if 0.45 <= number <= 0.55:
                return '中立'
            elif number > 0.55:
                return '正面'
            else:
                return '负面'
        except:
            return '中立'

    df = df.dropna(how='any',axis=0)
    df['情感类型'] = df['评论内容'].apply(snownlp_content)
    df['分词'] = df['评论内容'].apply(get_cut_words)
    df = df.dropna(subset=['分词'], axis=0)
    df.to_csv('./数据集/{}.csv'.format(name), encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    list_name = ['阿凡达2：水之道','速度与激情10','银河护卫队3','奥本海默','变形金刚7：超能勇士崛起','蜘蛛侠：纵横宇宙','蚁人3','芭比','超级马里奥','黑豹2','闪电侠','龙与地下城','雷霆沙赞','疯狂元素城','小美人鱼',
                 '阿凡达1','速度与激情9','银河护卫队2','信条','大黄蜂','蜘蛛侠：平行宇宙','蚁人2','黑豹','神奇女侠1984','疯狂动物城','美女与野兽','长津湖','战狼2','你好李焕英','哪吒之魔童降世','流浪地球','满江红',
                 '唐人街探案3','复仇者联盟4终局之战','长津湖之水门桥','流浪地球2','红海行动']

    for l in tqdm(list_name):
        main(l)


    # def list_files_in_directory(directory):
    #     return os.listdir(directory)
    #
    #
    # # 使用
    # folder_path = "数据集"
    # list_name1 = list_files_in_directory(folder_path)
    # list_name2 = []
    # for l in list_name1:
    #     l = str(l).replace('.csv','')
    #     list_name2.append(l)
    #
    # list_name3 = list(set(list_name).difference(list_name2))

    # # 使用 ThreadPoolExecutor
    # with concurrent.futures.ProcessPoolExecutor(max_workers=5)as e:
    #     # 通过 map function 来并发启动加载url的任务，并收集结果到future_to_url
    #     future_to_url = [e.submit(main, l) for l in list_name]
    #     for future in tqdm(concurrent.futures.as_completed(future_to_url),total=len(future_to_url)):
    #         print(futuer.result())



