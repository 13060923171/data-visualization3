import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

import random
from PIL import Image
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import networkx as nx
import codecs
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool
from scipy.sparse import coo_matrix
from tqdm import tqdm

def emotion_pie():
    df = pd.read_csv('sum_zh_data.csv')
    new_df = df['sentiment'].value_counts()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(9, 6), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('情感占比分布')
    plt.tight_layout()
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.savefig('情感占比分布.png')


def emotion_word(x):
    df = pd.read_csv('sum_zh_data.csv')
    df1 = df[df['sentiment'] == x]
    d = {}
    list_text = []
    for t in df1['fenci']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            list_text.append(i)
            d[i] = d.get(i, 0) + 1

    ls = list(d.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    y_data = []
    for key, values in ls[:100]:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_csv('{}-高频词Top100.csv'.format(x), encoding='utf-8-sig', index=False)

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl({}, 100%, 50%)".format(np.random.randint(0, 300))

    # 读取背景图片
    background_Image = np.array(Image.open('image.jpg'))
    text = ' '.join(list_text)
    wc = WordCloud(
        collocations=False,  # 禁用词组
        font_path='simhei.ttf',  # 中文字体路径
        margin=20,  # 词云图边缘宽度
        mask=background_Image,  # 背景图形
        scale=3,  # 放大倍数
        max_words=200,  # 最多词个数
        random_state=42,  # 随机状态
        width=800,  # 图片宽度
        height=600,  # 图片高度
        min_font_size=15,  # 最小字体大小
        max_font_size=90,  # 最大字体大小
        background_color='#ecf0f1',  # 背景颜色
        color_func=color_func  # 字体颜色函数
    )
    # 生成词云
    wc.generate_from_text(text)
    # 存储图像
    wc.to_file("{}-top100-词云图.png".format(x))


def net_analyze():
    # 读取数据
    df = pd.read_csv('sum_zh_data.csv')
    word = list(set(' '.join(df['fenci']).split()))  # 获取所有唯一词汇
    word_index = {w: i for i, w in enumerate(word)}  # 创建词汇索引字典

    # 构建词汇共现矩阵的辅助函数
    def process_line(line):
        line = line.strip()
        nums = line.split(' ')
        indices = [(word_index[nums[i]], word_index[nums[j]]) for i in range(len(nums)) for j in
                   range(i + 1, len(nums))]
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
    plt.figure(figsize=(20, 14), dpi=300)
    pos = nx.spring_layout(G, iterations=50)
    nx.draw_networkx_nodes(G, pos, alpha=0.7, node_size=800)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8, edge_color='g')
    nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1, font_size=16)
    plt.title("语义网络分析")
    plt.savefig('语义网络分析.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    emotion_pie()
    list1 = ['正面','负面']
    for l in list1:
        emotion_word(l)
    net_analyze()
