import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import os
from tqdm import tqdm
import ast  # 确保导入了 ast 模块

import paddle
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.data import Pad, Tuple

# 导入停用词列表
stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


#去掉标点符号，以及机械压缩
def preprocess_word(word):
    word1 = str(word)
    word1 = re.sub(r'#\w+#', '', word1)
    word1 = re.sub(r'【.*?】', '', word1)
    word1 = re.sub(r'@[\w]+', '', word1)
    word1 = re.sub(r'[a-zA-Z]', '', word1)
    word1 = re.sub(r'\.\d+', '', word1)
    return word1


def emjio_tihuan(x):
    x1 = str(x)
    x2 = re.sub('(\[.*?\])', "", x1)
    x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
    x4 = re.sub(r'\n', '', x3)
    return x4


# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def get_cut_words(content_series):
    try:
        # 对文本进行分词和词性标注
        words = pseg.cut(content_series)
        # 保存名词和形容词的列表
        nouns_and_adjs = []
        # 逐一检查每个词语的词性，并将名词和形容词保存到列表中
        for word, flag in words:
            #判断是否为名词或者形容词或者动词
            if flag in ['Ag','a','ad','an','Ng','n','v','nr','ns','nt','nz']:
                if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
                        # 如果是名词或形容词，就将其保存到列表中
                        nouns_and_adjs.append(word)
        if len(nouns_and_adjs) != 0:
            return ' '.join(nouns_and_adjs)
        else:
            return np.NAN
    except:
        return np.NAN

def emotion(text):
    # 对输入文本进行分词和编码
    inputs = tokenizer(text, max_seq_len=128)
    input_ids = paddle.to_tensor([inputs['input_ids']], dtype='int64')
    token_type_ids = paddle.to_tensor([inputs['token_type_ids']], dtype='int64')

    # 进行情感分类
    logits = model(input_ids, token_type_ids)

    # 获取分类结果
    probs = paddle.nn.functional.softmax(logits, axis=-1)
    probs = probs.numpy()[0]

    # 输出类别和对应的分值
    label = "正面" if probs[1] > probs[0] else "负面"

    # d = {"标签":label,"正面分值":f'{probs[1]:.4f}',"负面分值":f"{probs[0]:.4f}"}
    return label,probs[1],probs[0]


# 定义函数来解析和拆分数据
def parse_and_split(row):
    # 确保 row 是字符串形式
    if not isinstance(row, str):
        row = str(row)

    # 将字符串解析为字典
    player_dict = ast.literal_eval(row)

    # 提取玩家ID和服务器ID
    player_ids = list(player_dict.keys())
    server_ids = list(player_dict.values())

    # 构建结果字典
    result = {
        '标签': server_ids[0] if len(server_ids) > 0 else np.NAN,
        '正面分值': server_ids[1] if len(server_ids) > 0 else np.NAN,
        '负面分值': server_ids[2] if len(server_ids) > 0 else np.NAN,
    }
    return pd.Series(result)


# 加载预训练的 ERNIE-3.0 模型和分词器
model_name = "ernie-3.0-base-zh"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=2)

# 将模型设置为评估模式
model.eval()

list_name1 = ['25-31博文.xlsx']
list_name2 = ['25-31评论.xlsx']
list_name3 = ['25-31博文&评论']

list_lable = []
list_pos = []
list_neg = []

list_lable1 = []
list_pos1 = []
list_neg1 = []

for n1,n2,n3 in zip(list_name1,list_name2,list_name3):
    df = pd.read_excel(f'{n1}')
    df = df.drop_duplicates(subset=['发布内容'])
    df['发布内容'] = df['发布内容'].apply(preprocess_word)
    df['发布内容'] = df['发布内容'].apply(emjio_tihuan)
    df = df.dropna(subset=['发布内容'], axis=0)
    df['fenci'] = df['发布内容'].apply(get_cut_words)
    df = df.dropna(subset=['fenci'], axis=0)
    for l in tqdm(df['发布内容']):
        l1,l2,l3 = emotion(l)
        list_lable.append(l1)
        list_pos.append(l2)
        list_neg.append(l3)

    data = pd.DataFrame()
    data['原文本'] = df['发布内容']
    data['分词'] = df['fenci']
    data['类型'] = '博文'
    data['标签'] = list_lable
    data['正面分值'] = list_pos
    data['负面分值'] = list_neg


    df1 = pd.read_excel(f'{n2}')
    df1 = df1.drop_duplicates(subset=['评论文本'])
    df1['评论文本'] = df1['评论文本'].apply(preprocess_word)
    df1['评论文本'] = df1['评论文本'].apply(emjio_tihuan)
    df1 = df1.dropna(subset=['评论文本'], axis=0)
    df1['fenci'] = df1['评论文本'].apply(get_cut_words)
    df1 = df1.dropna(subset=['fenci'], axis=0)

    for l in tqdm(df1['评论文本']):
        l1,l2,l3 = emotion(l)
        list_lable1.append(l1)
        list_pos1.append(l2)
        list_neg1.append(l3)

    data1 = pd.DataFrame()
    data1['原文本'] = df1['评论文本']
    data1['分词'] = df1['fenci']
    data1['类型'] = '评论'
    data1['标签'] = list_lable1
    data1['正面分值'] = list_pos1
    data1['负面分值'] = list_neg1

    data2 = pd.concat([data,data1],axis=0)
    data2.to_excel(f'{n3}.xlsx',index=False)
