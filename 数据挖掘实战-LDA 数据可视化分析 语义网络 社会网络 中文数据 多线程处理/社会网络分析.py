import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

import numpy as np
import pandas as pd

import codecs
from scipy.sparse import coo_matrix
from tqdm import tqdm

# 读取词汇共现数据并构建DataFrame
data = []
with open('word_node.txt', 'r', encoding='utf-8') as f:
    for line in f:
        word1, word2, weight = line.strip().split()
        data.append([word1, word2, int(weight)])

df = pd.DataFrame(data, columns=['source', 'target', 'weight'])
df = df.sort_values(by='weight', ascending=False).iloc[:100]
df = df.drop(['weight'],axis=1)
# 构建社会网络
G = nx.Graph()

# 添加边来构建网络（假设CSV文件有source和target两列）
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    G.add_edge(row['source'], row['target'])

# 可视化社会网络
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(20, 14), dpi=300)
pos = nx.spring_layout(G, iterations=20)
nx.draw_networkx_nodes(G, pos, alpha=0.7, node_size=800)
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8, edge_color='g')
nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1, font_size=10)
plt.title("社会网络分析")
plt.savefig('社会网络分析.png', bbox_inches='tight')
plt.show()