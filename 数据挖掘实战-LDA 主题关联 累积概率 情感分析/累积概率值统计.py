import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from collections import defaultdict


import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


def lad_probability(name,number):
    df = pd.read_csv(f'{name}.csv')

    train = []
    stop_word = []
    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_word.append(line.strip())
    for line in df['fenci']:
        line = [str(word).strip(' ') for word in line.split(' ') if len(word) >= 2 and word not in stop_word]
        train.append(line)

    # 构建为字典的格式
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]

    # 训练LDA模型
    lda = LdaModel(corpus, num_topics=number, id2word=dictionary, random_state=42)

    # 获取每个主题和每个词在各主题下的概率分布

    word_topic = pd.read_excel(f'./{name}/主题词分布表.xlsx')
    word_topic = word_topic.iloc[:30]

    list_columns = []
    for c in word_topic.columns:
        list_columns.append(c)
    # 初始化一个默认字典来存储不同维度的词组
    topic_word_groups = defaultdict(list)

    # 遍历数据框，根据维度分类词组
    for _, row in word_topic.iterrows():
        for i in range(0, len(list_columns), 3):
            word_col = list_columns[i]
            dimension_col = list_columns[i + 2]
            if pd.notnull(row[dimension_col]):
                topic_word_groups[row[dimension_col]].append(row[word_col])


    # 将 defaultdict 转换为字典
    topic_word_groups = dict(topic_word_groups)
    topic_word_groups1 = {}
    labels = []
    # 打印结果
    for dimension, words in topic_word_groups.items():
        if '1' not in dimension and '2' not in dimension and '3' not in dimension and '4' not in dimension and '5' not in dimension and '6' not in dimension and '7' not in dimension and '8' not in dimension:
            topic_word_groups1[dimension] = words
            labels.append(dimension)


    # 获取主题-词矩阵(即topic-term matrix)，其中每个元素是词在每个主题下的概率
    topic_term_matrix = lda.get_topics()

    # 存储累积概率
    cumulative_prob = {topic: [0] * lda.num_topics for topic in topic_word_groups1}

    # 计算每个词组在每个主题下的累积概率
    for topic, words in topic_word_groups1.items():
        for word in words:
            if word in dictionary.token2id:
                word_id = dictionary.token2id[word]
                for topic_id in range(lda.num_topics):
                    cumulative_prob[topic][topic_id] += topic_term_matrix[topic_id, word_id]
            else:
                print(f"Warning: Word '{word}' not in dictionary!")


    # 将累积概率结果转换为 DataFrame
    df = pd.DataFrame(cumulative_prob)
    df.index = [f'Topic {i+1}' for i in range(len(df))]
    df = df.T.reset_index()  # 转置并重置索引
    df.columns = ['Dimension'] + [f'Topic {i+1}' for i in range(len(df.columns) - 1)]
    # 将 DataFrame 保存为 Excel 文件
    excel_file = f'./{name}/累计概率值.xlsx'
    df.to_excel(excel_file, index=False)

    # 打印累积概率
    for topic, probs in cumulative_prob.items():
        print(f"{topic} cumulative probabilities:")
        for topic_id, prob in enumerate(probs):
            print(f"  Topic {topic_id + 1}: {prob:.4f}")


    # 可视化累积概率
    x = np.arange(number)  # 主题的数量
    fig, ax = plt.subplots(figsize=(18, 10),dpi=500)

    # 设置不同的颜色
    colors = sns.color_palette("hsv", len(labels))

    # 绘制柱状图
    for idx, word in enumerate(labels):
        ax.bar(x + (idx * 0.15), cumulative_prob[word], width=0.15, color=colors[idx], label=word)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 添加标签和标题
    ax.set_xlabel('Topics')
    ax.set_ylabel('Cumulative Probabilities')
    ax.set_title('Cumulative Probabilities of Words Across Topics')
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels([f'Topic {i}' for i in range(number)])
    ax.legend(title='Words')
    plt.savefig(f'./{name}/累计概率值.png')
    plt.show()


if __name__ == '__main__':
    list_name = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
    list_number = [16,18,14,17,18,15,19,19,13,14,16,16]
    for name,number in zip(list_name,list_number):
        lad_probability(name,number)