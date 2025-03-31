import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from scipy.spatial.distance import cosine
from collections import defaultdict
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

df1 = pd.read_csv('1月.csv')
df2 = pd.read_csv('2月.csv')
df3 = pd.read_csv('3月.csv')
df4 = pd.read_csv('4月.csv')
df5 = pd.read_csv('5月.csv')
df6 = pd.read_csv('6月.csv')
df7 = pd.read_csv('7月.csv')
df8 = pd.read_csv('8月.csv')
df9 = pd.read_csv('9月.csv')
df10 = pd.read_csv('10月.csv')
df11 = pd.read_csv('11月.csv')
df12 = pd.read_csv('12月.csv')

# 示例文本语料
quarters = {
    'Jan': df1['fenci'].tolist(),
    'Feb': df2['fenci'].tolist(),
    'Mar': df3['fenci'].tolist(),
    'Apr': df4['fenci'].tolist(),
    'May': df5['fenci'].tolist(),
    'Jun': df6['fenci'].tolist(),
    'Jul': df7['fenci'].tolist(),
    'Aug': df8['fenci'].tolist(),
    'Sept': df9['fenci'].tolist(),
    'Oct': df10['fenci'].tolist(),
    'Nov': df11['fenci'].tolist(),
    'Dec': df12['fenci'].tolist(),
}


# 指定每个季度的主题数量
num_topics_per_quarter = {'Jan': 9, 'Feb': 13, 'Mar': 13, 'Apr': 8,'May': 9, 'Jun': 8, 'Jul': 8, 'Aug': 12,'Sept': 16, 'Oct': 14, 'Nov': 9, 'Dec': 9}

# 构建全局词汇表
all_texts = []
for texts in quarters.values():
    all_texts.extend(texts)

processed_texts = [[word for word in document.lower().split()] for document in all_texts]
global_dictionary = corpora.Dictionary(processed_texts)

# 预处理和训练LDA模型
lda_models = {}
topic_term_matrices = {}

for quarter, texts in quarters.items():
    processed_texts = [[word for word in document.lower().split()] for document in texts]

    # 创建局部词典和语料库
    local_dictionary = corpora.Dictionary(processed_texts)
    corpus = [global_dictionary.doc2bow(text) for text in processed_texts]

    # 训练LDA模型
    num_topics = num_topics_per_quarter[quarter]
    lda = LdaModel(corpus, num_topics=num_topics, id2word=global_dictionary, random_state=42)
    lda_models[quarter] = lda

    # 获取主题-词矩阵
    topic_term_matrix = lda.get_topics()
    topic_term_matrices[quarter] = topic_term_matrix


# 计算相似度函数
def compute_cosine_similarity(matrix1, matrix2):
    similarity_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))
    for i in range(matrix1.shape[0]):
        for j in range(matrix2.shape[0]):
            similarity_matrix[i, j] = 1 - cosine(matrix1[i], matrix2[j])
    return similarity_matrix


# 计算相邻季度的相似度
similarities = {}
quarters_keys = list(quarters.keys())
for i in range(len(quarters_keys) - 1):
    q1 = quarters_keys[i]
    q2 = quarters_keys[i + 1]
    similarity_matrix = compute_cosine_similarity(topic_term_matrices[q1], topic_term_matrices[q2])
    similarities[(q1, q2)] = similarity_matrix

    # 可视化相邻季度的主题相似度矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm',
                xticklabels=[f'{q2} Topic {i}' for i in range(similarity_matrix.shape[1])],
                yticklabels=[f'{q1} Topic {i}' for i in range(similarity_matrix.shape[0])])
    plt.title(f'Topic Similarity between {q1} and {q2}')
    plt.show()

# 筛选强关联关系
strong_relations = {}
threshold = 0.5  # 设置相似度阈值
for (q1, q2), similarity_matrix in similarities.items():
    relations = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if similarity_matrix[i, j] > threshold:
                relations.append((i, j, similarity_matrix[i, j]))
    strong_relations[(q1, q2)] = relations

data = pd.DataFrame()
data['Pre-Theme'] = ['Pre-Theme']
data['Theme'] = ['Theme']
data['Sim'] = ['Sim']
data.to_csv('The_Calculation_of_Subject_Relationship.csv',mode='w',index=False,encoding='utf-8-sig',header=False)
# 输出强关联关系结果
for (q1, q2), relations in strong_relations.items():
    print(f"Strong relations between {q1} and {q2} (similarity > {threshold}):")
    for rel in relations:
        print(f"  {q1} Topic {rel[0]} and {q2} Topic {rel[1]} with similarity {rel[2]:.2f}")
        data['Pre-Theme'] = [f'{q1} Topic {rel[0]}']
        data['Theme'] = [f'{q2} Topic {rel[1]}']
        data['Sim'] = [f'{rel[2]:.2f}']
        data.to_csv('The_Calculation_of_Subject_Relationship.csv', mode='a+', index=False, encoding='utf-8-sig',
                    header=False)


def create_sankey_data(relations):
    nodes = []
    links = {"source": [], "target": [], "value": []}
    node_indices = {}
    idx = 0

    for (q1, q2), rel_list in relations.items():
        for (topic1, topic2, similarity) in rel_list:
            source_node = f'{q1} Topic {topic1}'
            target_node = f'{q2} Topic {topic2}'

            if source_node not in node_indices:
                node_indices[source_node] = idx
                nodes.append(source_node)
                idx += 1

            if target_node not in node_indices:
                node_indices[target_node] = idx
                nodes.append(target_node)
                idx += 1

            links["source"].append(node_indices[source_node])
            links["target"].append(node_indices[target_node])
            links["value"].append(similarity)

    return nodes, links


node_labels, sankey_links = create_sankey_data(strong_relations)

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_labels
    ),
    link=sankey_links
)])

fig.update_layout(title_text="Theme Relevance", font_size=10)
# 将桑基图保存为HTML文件
fig.write_html("Theme_Relevance.html")
fig.show()
