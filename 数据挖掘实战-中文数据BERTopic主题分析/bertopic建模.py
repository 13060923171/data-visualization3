import numpy as np
import pandas as pd
from bertopic import BERTopic
import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese').to(device)  # 将模型移动到GPU

# 读取数据并进行预处理
df = pd.read_excel('data_v6.xlsx')

def data_process(x):
    x1 = str(x).strip(" ").strip("\n").split(" ")
    if len(x1) >= 2:
        return str(x).strip(" ").strip("\n")
    else:
        return np.NAN

df['分词'] = df['分词'].apply(data_process)
df = df.dropna(subset=['分词'], axis=0)

# 打印预处理后的文档，检查是否包含有效词汇
documents = df['分词'].tolist()

# 确保文档列表中没有空字符串
documents = [doc for doc in documents if isinstance(doc, str) and doc.strip() != ""]

# 批量处理函数
def generate_embeddings(doc_batch):
    inputs = tokenizer(doc_batch,
                       padding='max_length',
                       max_length=256,  # 适当增大 max_length 以提高处理效率
                       truncation=True,
                       return_tensors="pt").to(device)  # 将输入数据移动到GPU

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # 将输出从GPU移回CPU并转为numpy数组
    return embeddings

# 并行处理函数
def parallel_embeddings(doc_batches):
    all_embeddings = Parallel(n_jobs=-1)(
        delayed(generate_embeddings)(batch) for batch in tqdm(doc_batches, desc="生成嵌入"))
    return np.vstack(all_embeddings)

# 将文档分成批次
batch_size = 16  # 增大 batch size 以提高处理效率
doc_batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

# 并行生成嵌入
embeddings = parallel_embeddings(doc_batches)

# 创建CountVectorizer模型
vectorizer_model = CountVectorizer()

# 初始化 BERTopic 模型
topic_model = BERTopic(vectorizer_model=vectorizer_model, embedding_model=model)

# 使用自定义嵌入进行主题提取
try:
    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

    # 创建保存目录
    if not os.path.exists('./bertopic_data'):
        os.makedirs('./bertopic_data')

    # 输出主题信息
    topic_docs = topic_model.get_document_info(documents)
    topic_docs.to_csv('./bertopic_data/聚类结果.csv', encoding='utf-8-sig', index=False)

    # 可视化主题词分布
    fig_topics = topic_model.visualize_barchart()
    fig_topics.write_html("./bertopic_data/topics_barchart.html")

    data = pd.DataFrame()
    data['主题'] = ['主题']
    data['主题词'] = ['主题词']
    data.to_csv('./bertopic_data/主题词.csv', encoding='utf-8-sig', index=False, header=False, mode='w')

    # 打印每个主题的关键词
    for topic in topic_model.get_topics().keys():
        print(f"主题 {topic}: {topic_model.get_topic(topic)}")
        data = pd.DataFrame()
        data['主题'] = [topic]
        data['主题词'] = [topic_model.get_topic(topic)]
        data.to_csv('./bertopic_data/主题词.csv', encoding='utf-8-sig', index=False, header=False, mode='a+')
except ValueError as e:
    print("错误信息：", e)