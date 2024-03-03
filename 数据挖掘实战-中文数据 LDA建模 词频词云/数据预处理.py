from PyPDF2 import PdfReader
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import re
import jieba
import jieba.posseg as pseg

# 读取 PDF 文件
def extract_text_from_pdf(pdf_path):
    text = ''
    pdf = PdfReader(pdf_path)
    for page_num in range(len(pdf.pages)):
        text += pdf.pages[page_num].extract_text()
    text = text.replace("\n", "").strip(" ")
    return text


# 将提取出的内容保存到 Excel 表格
def save_to_excel(text):
    df = pd.DataFrame()
    df['content'] = text
    df.to_excel('data.xlsx',index=False)


def list_files_in_folder(folder_path):
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_names.append(file)
    return file_names

def main1():
    # 导入停用词列表
    stop_words = []
    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    # 去掉标点符号，以及机械压缩
    def preprocess_word(word):
        word1 = str(word)
        # word1 = re.sub(r'转发微博', '', word1)
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

    def get_cut_words(content_series):
        try:
            # 对文本进行分词和词性标注
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

    df = pd.read_excel('data.xlsx')

    df['content'] = df['content'].apply(preprocess_word)
    df['content'] = df['content'].apply(emjio_tihuan)
    df.dropna(subset=['content'], axis=0, inplace=True)
    df['fenci'] = df['content'].apply(get_cut_words)
    new_df = df.dropna(subset=['fenci'], axis=0)
    new_df.to_excel('data.xlsx', index=False)


def main2():
    df1 = pd.read_excel('data.xlsx')
    df2 = pd.read_excel('科创板IPO申报公司信息.xls',sheet_name='科创板IPO申报公司信息 (2)')
    list_type = []

    def zidian_value(content):
        for i,j in zip(df2['发行人名称'],df2['所属领域']):
            if i in content:
                return j

    for i in df1['content']:
        try:
            content = zidian_value(str(i))
            list_type.append(content)
        except:
            list_type.append('无')

    df1['所属领域'] = list_type
    df1.to_excel('new_data.xlsx',index=False)


if __name__ == '__main__':
    # folder_path = 'data'
    # files = list_files_in_folder(folder_path)
    # list_content = []
    # for file in tqdm(files):
    #     pdf_path = './data/{}'.format(file)
    #     content = extract_text_from_pdf(pdf_path)
    #     list_content.append(content)
    # save_to_excel(list_content)
    # main1()
    main2()
