
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端

import numpy as np
import pandas as pd
import networkx as nx
import codecs
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool
from scipy.sparse import coo_matrix
from tqdm import tqdm


def main1(df,name):
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
    df = df.sort_values(by='weight', ascending=False).iloc[:80]

    # 构建语义网络
    edges = list(zip(df['word1'], df['word2']))
    G = nx.Graph()
    G.add_edges_from(edges)

    # # 只考虑最大的连通分量
    # largest_connected_component = max(nx.connected_components(G), key=len)
    # G_lcc = G.subgraph(largest_connected_component)
    #
    # # 计算网络指标
    # density = nx.density(G)  # 密度值
    # average_shortest_path_length = nx.average_shortest_path_length(G_lcc)  # 平均距离值（只考虑最大连通分量）
    # degree_centrality = nx.degree_centrality(G)  # 点度中心度
    # betweenness_centrality = nx.betweenness_centrality(G)  # 中介中心度

    # # 对点度中心度和中介中心度进行归一化
    # max_degree = max(degree_centrality.values())
    # min_degree = min(degree_centrality.values())
    # normalized_degree_centrality = {node: (centrality - min_degree) / (max_degree - min_degree) for node, centrality in
    #                                 degree_centrality.items()}
    #
    # max_betweenness = max(betweenness_centrality.values())
    # min_betweenness = min(betweenness_centrality.values())
    # normalized_betweenness_centrality = {node: (centrality - min_betweenness) / (max_betweenness - min_betweenness) for
    #                                      node, centrality in betweenness_centrality.items()}
    #
    # # 将指标保存到文件
    # with codecs.open("network_metrics.txt", "w", "utf-8") as metrics_file:
    #     metrics_file.write(f"网络密度: {density}\n")
    #     metrics_file.write(f"平均最短路径长度 (最大连通分量): {average_shortest_path_length}\n")
    #     metrics_file.write("点度中心度:\n")
    #     for node, centrality in degree_centrality.items():
    #         metrics_file.write(f"{node}: {centrality}\n")
    #     metrics_file.write("中介中心度:\n")
    #     for node, centrality in betweenness_centrality.items():
    #         metrics_file.write(f"{node}: {centrality}\n")
    #     metrics_file.write("归一化点度中心度:\n")
    #     for node, centrality in normalized_degree_centrality.items():
    #         metrics_file.write(f"{node}: {centrality}\n")
    #     metrics_file.write("归一化中介中心度:\n")
    #     for node, centrality in normalized_betweenness_centrality.items():
    #         metrics_file.write(f"{node}: {centrality}\n")


    # 可视化语义网络
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(20, 9),dpi=300)
    pos = nx.spring_layout(G, iterations=30)
    nx.draw_networkx_nodes(G, pos, alpha=0.7, node_size=800)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8, edge_color='g')
    nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1, font_size=10)
    plt.title(f"{name}网络分析")
    plt.savefig(f'{name}网络分析.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    df = pd.read_excel('new_data.xlsx')
    main1(df, '')