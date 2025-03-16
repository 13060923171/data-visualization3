import pandas as pd
import re
import numpy as np
from scipy.stats import pearsonr

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

import networkx as nx
from collections import Counter
from itertools import combinations

import networkx as nx
import codecs
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool
from scipy.sparse import coo_matrix
from tqdm import tqdm
from itertools import product
from scipy.stats import pearsonr
import random
from PIL import Image
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# 读取数据（假设包含'fenci'和'sentiment'列）
df = pd.read_csv('new_data.csv')

# 初始化参数
brands = ["小米", "特斯拉"]
feature_list = ["雷总", "汽车","手机","雷军","万辆","售后","卫生巾","生产","电车","营销","油车","交付","工厂","员工","马斯克","广告","服务","期待","质量","雷神","公司","军哥","米粉","电池","产品","造车","红米"]  # 需替换为实际特性词典

def data_process(x):
    x1 = str(x)
    if  x1  in ["雷总", "汽车","手机","雷军","万辆","售后","卫生巾","生产","电车","营销","金山","布斯","手机","米粉","友商","螺丝","布斯","军儿","流量","营销","雷","米","红米","造车"] :
        return "小米"
    elif  x1 in ["电动","自动驾驶","科技","加速","Model 3","Model Y","充电桩","Autopilot","马斯克", "特斯拉", "电池", "屏幕"]:
        return "特斯拉"
    elif ("雷" in x1 and "斯克" in x1) or  ("小米" in x1 and "特斯拉" in x1) or ("布斯" in x1 and "斯克" in x1):
        return "品牌共现"
    else:
        return "其他"

df['品牌'] = df['fenci'].apply(data_process)

df.to_csv('new_data.csv',index=False)

# 辅助函数（处理正则匹配）
def contains_keyword(text, word):
    return bool(re.search(rf'\b{re.escape(word)}\b', str(text)))


# ==================== 品牌共现分析 ====================
# 共现矩阵初始化
brand_matrix = np.zeros((len(brands), len(brands)))

# 构建共现矩阵
for _, row in df.iterrows():
    mentioned = [i for i, b in enumerate(brands) if contains_keyword(row['fenci'], b)]
    for i in mentioned:
        for j in mentioned:
            if i < j:  # 避免重复计数
                brand_matrix[i][j] += 1

print("品牌共现矩阵：\n", pd.DataFrame(brand_matrix, index=brands, columns=brands))

data = pd.DataFrame(brand_matrix, index=brands, columns=brands)
data.to_csv('品牌共现矩阵.csv', encoding='utf-8-sig', index=False)
# 共现率计算
brand_counts = np.array([sum(df['fenci'].apply(lambda x: contains_keyword(x, b))) for b in brands])
total = len(df)

cooccur_rates = np.zeros_like(brand_matrix, dtype=float)
for i in range(len(brands)):
    for j in range(len(brands)):
        if i != j:
            p_ab = brand_matrix[i, j] / total
            p_a = brand_counts[i] / total
            p_b = brand_counts[j] / total
            denominator = p_a + p_b - p_ab
            cooccur_rates[i, j] = p_ab / denominator if denominator else 0

print("\n共现率矩阵：\n", pd.DataFrame(cooccur_rates, index=brands, columns=brands))

data1 = pd.DataFrame(cooccur_rates, index=brands, columns=brands)
data1.to_csv('共现率矩阵.csv', encoding='utf-8-sig', index=False)

# 情感相关性分析
sentiment_map = {'正面': 0, '中性': 1, '负面': 2}
sentiment_matrix = np.zeros((len(brands), len(brands), 3))


for _, row in df.iterrows():
    mentioned = [i for i, b in enumerate(brands) if contains_keyword(row['fenci'], b)]
    sen_idx = sentiment_map.get(row['sentiment'], 1)
    for i in mentioned:
        for j in mentioned:
            if i < j:
                sentiment_matrix[i, j, sen_idx] += 1


# 计算相关系数
flat_cooccur = brand_matrix.flatten()
flat_sentiment = sentiment_matrix.sum(axis=2).flatten()
correlation, _ = pearsonr(flat_cooccur, flat_sentiment)
print(f"\n情感相关性系数：{correlation:.3f}")

# ==================== 特性共现分析 ====================
# 品牌-特性矩阵
bf_matrix = np.zeros((len(brands), len(feature_list)))

for b_idx, brand in enumerate(brands):
    for f_idx, feature in enumerate(feature_list):
        bf_matrix[b_idx, f_idx] = sum(
            contains_keyword(text, brand) and contains_keyword(text, feature)
            for text in df['fenci']
        )

print("\n品牌-特性矩阵：\n", pd.DataFrame(bf_matrix, index=brands, columns=feature_list))


data2 = pd.DataFrame(bf_matrix, index=brands, columns=feature_list)
data2.to_csv('品牌-特性矩阵.csv', encoding='utf-8-sig', index=False)

# 网络密度计算
actual_edges = np.count_nonzero(bf_matrix)
max_edges = bf_matrix.size
density = actual_edges / max_edges
print(f"\n网络密度：{density:.2%}")


# 时间变化分析（示例按评论顺序分3个时段）
periods = np.array_split(df, 3)
change_rates = []

for i in range(1, len(periods)):
    prev = np.sum([contains_keyword(text, b) and contains_keyword(text, f)
                   for b in brands for f in feature_list
                   for text in periods[i - 1]['fenci']])

    curr = np.sum([contains_keyword(text, b) and contains_keyword(text, f)
                   for b in brands for f in feature_list
                   for text in periods[i]['fenci']])

    change_rates.append((curr - prev) / prev if prev else 0)

print("\n时段变化率：", [f"{x:.1%}" for x in change_rates])



#++++++++++++++++++++可视化部分+++++++++++++++++++++++
#网络语义图
def main1(df):
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
    plt.figure(figsize=(16, 9),dpi=300)
    pos = nx.spring_layout(G, iterations=100,k=0.5,seed=42)
    nx.draw_networkx_nodes(G, pos, alpha=0.7, node_size=800)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.8, edge_color='g')
    nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=1, font_size=10)
    plt.title(f"网络分析")
    plt.savefig(f'网络分析.png', bbox_inches='tight')
    plt.show()

def sentiment_pie(df,name):
    # 情感分析饼图
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    df['sentiment'].value_counts().plot.pie(
        autopct='%1.1f%%',
        colors=['#e3716e', '#eca680', '#7ac7e2'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    plt.title(f'{name}_情感分布分析', fontsize=14, pad=20)
    plt.savefig(f'{name}_情感分布分析.png', dpi=300)


def pp_bar(x_data,y_data):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_data, y_data, color=['#e3716e', '#eca680', '#7ac7e2'], edgecolor='white')
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.,
                height + 0.3,
                f'{height}',
                ha='center',
                va='bottom',
                fontsize=12,
                color='black')

    # 图表美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('品牌名称', fontsize=12, labelpad=10)
    ax.set_ylabel('提及频次', fontsize=12, labelpad=10)
    ax.set_title('品牌提及频次分析', fontsize=14, pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('品牌提及频次.png', dpi=300, bbox_inches='tight')


def word_fx(df,name):
    d = {}
    list_text = []
    for t in df['fenci']:
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
    data['frequency'] = round(data['counts'] / data['counts'].sum(),6)

    data.to_csv(f'{name}-高频词.csv', encoding='utf-8-sig', index=False)

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        return "hsl({}, {}%, {}%)".format(np.random.randint(206, 325),np.random.randint(34, 64),np.random.randint(33, 55))

    # 读取背景图片
    background_Image = np.array(Image.open('images.png'))
    text = ' '.join(list_text)
    wc = WordCloud(
        collocations=False,  # 禁用词组
        font_path='simhei.ttf',  # 中文字体路径
        margin=3,  # 词云图边缘宽度
        mask=background_Image,  # 背景图形
        scale=3,  # 放大倍数
        max_words=150,  # 最多词个数
        random_state=42,  # 随机状态
        width=800,  # 图片宽度
        height=600,  # 图片高度
        min_font_size=15,  # 最小字体大小
        max_font_size=90,  # 最大字体大小
        background_color='#fdfefe',  # 背景颜色
        color_func=color_func  # 字体颜色函数
    )
    # 生成词云
    wc.generate_from_text(text)
    # 存储图像
    wc.to_file(f'{name}-词云图.png')


if __name__ == '__main__':

    new_df = df['品牌'].value_counts()
    new_df = new_df[1:]
    x_data = ['小米', '特斯拉', '品牌共现']
    y_data = [845,677,1197]
    pp_bar(x_data,y_data)


    list_name = ['小米', '特斯拉', '品牌共现']
    for n in list_name:
        df1 = df[df['品牌'] == n]
        sentiment_pie(df1, n)

    # main1(df)

    list_name = ["其他",'小米', '特斯拉', '品牌共现']
    for n in list_name:
        df1 = df[df['品牌'] == n]
        word_fx(df1, n)
