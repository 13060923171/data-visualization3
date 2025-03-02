import pandas as pd
import numpy as np
import re
import MeCab
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 停用词列表
stopwords = []
with open("stopwords-ja.txt", 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f]

def clean_japanese_text(text):
    """数据清洗函数"""
    # 移除emoji和特殊符号
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    # 保留日文字符和标点
    jp_pattern = re.compile(
        r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F'
        r'！？。、・’"＠＃％＆・（）「」【】『』｛｝…‥―〜ー～]'
    )
    cleaned_text = jp_pattern.sub(' ', text)
    return re.sub(r'\s+', ' ', cleaned_text).strip()


def mechanical_compress(text):
    """机械压缩重复字符"""
    # 压缩连续重复的非字母数字字符（保留日文字符）
    return re.sub(r'([^ぁ-んァ-ン一-龥0-9a-zA-Z])\1+', r'\1', text)


def contains_japanese(text):
    """判断是否包含日文字符"""
    return re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text) is not None


def process_text(text):
    """完整的文本处理流程"""
    # 数据清洗
    cleaned = clean_japanese_text(text)
    if not cleaned:
        return None

    # 机械压缩
    compressed = mechanical_compress(cleaned)

    words = []
    tagger = MeCab.Tagger('-Ochasen')
    parsed = tagger.parse(compressed).split("\n")
    # # 分词和词性过滤
    for i in parsed:
        if i == 'EOS' or i == '':
            continue
        feature = i.split("\t")[3]
        surface = i.split("\t")[0]
        if "名詞" in feature or "形容" in feature or "組織名" in feature or "人名" in feature or "地名" in feature or "動詞" in feature or "副詞" in feature or "未知語" in feature or "タダ" in feature or "感動詞" in feature:
            if surface not in stopwords and not re.match(r'[、。．，,\.\s]+', surface) and len(surface) >= 2:
                words.append(surface)
    if len(words) != 0:
        return ' '.join(words)
    else:
        return np.NAN


def main1(cleaned_text1):
    # 情感分析（使用日语BERT模型）
    cleaned_text1 = str(cleaned_text1)
    result = sentiment_analyzer(cleaned_text1)
    label_id = int(result[0]['label'].split('_')[-1])
    label = LABEL_MAP[label_id]
    # score = result[0]['score']

    return label


if __name__ == '__main__':
    # # 读取数据
    # df = pd.read_excel('data.xlsx').dropna(subset=['content']).drop_duplicates(subset=['content'])
    #
    # # 处理数据
    # processed_data = []
    # for content in tqdm(df['content'], desc="Processing"):
    #     processed = process_text(content)
    #     if processed:
    #         processed_data.append({
    #             'original_content': content,
    #             'processed_content': processed
    #         })
    #
    # # 保存结果
    # result_df = pd.DataFrame(processed_data)
    # result_df.to_csv('processed_data.csv', index=False, encoding='utf-8-sig')

    # pip uninstall unidic_lite
    # pip install unidic_lite


    result_df = pd.read_csv('processed_data.csv')
    # 初始化模型和分词器
    # 标签映射
    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
    model_name = "cl-tohoku/bert-base-japanese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 创建情感分析管道
    sentiment_analyzer = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

    list_label = []
    for i in result_df['original_content']:
        label = main1(i)
        list_label.append(label)
    result_df['label'] = list_label
    result_df.to_csv('new_processed_data.csv', index=False)
    print("处理完成，结果已保存至 processed_data.csv")




