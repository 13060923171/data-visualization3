import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

import numpy as np
import pandas as pd

import codecs
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool
from scipy.sparse import coo_matrix
from tqdm import tqdm


# 读取数据
df = pd.read_csv('new_data.csv')  # 假设数据保存在your_data.csv中
word = list(set(' '.join(df['fenci']).split()))  # 获取所有唯一词汇
word_index = {w: i for i, w in enumerate(word)}  # 创建词汇索引字典

# 构建词汇共现矩阵的辅助函数
def process_line(line):
    line = line.strip()
    nums = line.split(' ')
    indices = [(word_index[nums[i]], word_index[nums[j]]) for i in range(len(nums)) for j in range(i + 1, len(nums))]
    return indices

# 多线程处理
results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_line, line) for line in df['fenci']]
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())


# 构建稀疏矩阵
rows, cols = zip(*[pair for sublist in results for pair in sublist])
data = np.ones(len(rows), dtype=np.int8)
word_vector = coo_matrix((data, (rows, cols)), shape=(len(word), len(word))).toarray()


# 保存词汇共现数据
with codecs.open("word_node.txt", "w", "utf-8") as words:
    for i in tqdm(range(len(word))):
        for j in range(i + 1, len(word)):
            if word_vector[i][j] > 0:
                words.write(f"{word[i]} {word[j]} {word_vector[i][j]}\r\n")

# 读取词汇共现数据并构建DataFrame
data = []
with open('word_node.txt', 'r', encoding='utf-8') as f:
    for line in f:
        word1, word2, weight = line.strip().split()
        data.append([word1, word2, int(weight)])

df = pd.DataFrame(data, columns=['word1', 'word2', 'weight'])
df = df.sort_values(by='weight', ascending=False).iloc[:100]

# 构建语义网络
edges = list(zip(df['word1'], df['word2']))
G = nx.Graph()
G.add_edges_from(edges)

# 可视化语义网络
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(20, 14),dpi=300)
pos = nx.spring_layout(G, iterations=50)
nx.draw_networkx_nodes(G, pos, alpha=0.7, node_size=800)
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8, edge_color='g')
nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1, font_size=10)
plt.title("语义网络分析")
plt.savefig('语义网络分析.png', bbox_inches='tight')
plt.show()
