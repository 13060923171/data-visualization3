import pandas as pd
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import torch
from tqdm import tqdm



# 读取数据并清理
data = pd.read_excel('post.xlsx')
data = data.drop_duplicates(subset=['发布内容'])
data = data.dropna(subset=['发布内容'], axis=0)

# 按日期分组并采样
data['发布时间'] = pd.to_datetime(data['发布时间'])
df = (
    data.groupby(data['发布时间'].dt.date)
      .apply(lambda x: x.sample(n=300) if len(x) >= 300 else x)
      .reset_index(drop=True)
)
df['发布内容'] = df['发布内容'].astype('str')

device = 0 if torch.cuda.is_available() else -1  # 使用GPU（如果可用）
# 使用更高效的模型和GPU
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)

# 并行化处理文本
def classify_text(text):
    return classifier(text)

texts = df['发布内容'].tolist()
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(classify_text, texts))

list_label = []
list_score = []
for result in tqdm(results):
    label = result[0]['label']
    score = result[0]['score']
    list_label.append(label)
    list_score.append(score)

# 添加结果到DataFrame
df['情感类别'] = list_label
df['情感得分'] = list_score


df.to_excel('new_data.xlsx', index=False)



