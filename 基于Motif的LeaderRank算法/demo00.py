import pandas as pd
import networkx as nx


# 定义Motif-based LeaderRank函数
# 认为edges是一个元组列表，表示每个连接的两个节点
# nodes是整个图中所有节点的列表
def leader_rank(G):
    # G是一个NetworkX图
    nodes = G.nodes()
    num_nodes = len(nodes)
    ground_node = num_nodes
    # 添加一个地面节点，与所有节点相连
    G.add_node(ground_node)
    G.add_edges_from([(ground_node, node) for node in nodes])
    # 结点的初始化得分
    scores = dict.fromkeys(nodes, 1.0)
    # MLR的循环过程，可以自定义迭代次数
    for _ in range(100):
        for i in nodes:
            scores[i] = 1 + 0.85*sum([scores[n] for n in G.predecessors(i)])
    # 删除地面结点
    G.remove_node(ground_node)
    del scores[ground_node]

    return scores

# 加载数据
df = pd.read_excel('data.xlsx', converters={'发布者ID': str})
#节点的影响力可以用点赞、转发和评论的总数
df['edge'] = df['转发'] + df['点赞'] + df['评论']

# 初始化一个有向图
G = nx.DiGraph()

# 根据数据构建图
for idx, row in df.iterrows():
    G.add_node(row['发布者ID'])
    if pd.notnull(row['edge']):
        G.add_edge(row['edge'], row['发布者ID'])

# 应用LeaderRank函数
scores = leader_rank(G)
# 将结果转为DataFrame以便更好处理
scores_df = pd.DataFrame(scores.items(), columns=['发布者ID', 'score'])

# 排序并获取得分最高的10个用户
top10 = scores_df.sort_values(by='score', ascending=False).head(10)

print(top10)