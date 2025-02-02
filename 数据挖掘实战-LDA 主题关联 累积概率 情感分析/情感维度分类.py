import numpy as np
from collections import defaultdict
from collections import Counter

import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType


def lad_probability(name):
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
        # if '1' not in dimension and '2' not in dimension and '3' not in dimension and '4' not in dimension and '5' not in dimension and '6' not in dimension and '7' not in dimension and '8' not in dimension:
        topic_word_groups1[dimension] = list(set(words))
        labels.append(dimension)


    list_type = []
    list_content = []
    for key,values in topic_word_groups1.items():
        list_type.append(key)
        list_content.append(values)

    def emotion_type(word):
        try:
            words = word.split()
            # 用于记录每个词语及其出现的列表
            matches_with_lists = []
            # 遍历每个词语，并找出它出现在哪个列表
            for word in words:
                found_lists = [f"{list_type[i + 1]}" for i, lst in enumerate(list_content) if word in lst]
                if found_lists:
                    matches_with_lists.append(found_lists[0])

            # 使用 Counter 统计每个词语出现的次数
            word_counts = Counter(matches_with_lists)
            # 找出出现次数最多的词语
            most_common_word = word_counts.most_common(1)
            return most_common_word[0][0]
        except:
            return np.NAN

    df = pd.read_csv(f'./{name}/{name}_data.csv')
    df['维度'] = df['fenci'].apply(emotion_type)
    list_dimensions = []
    for i in df['维度'].value_counts().index[:3]:
        list_dimensions.append(i)

    list_emotion = []
    for m in list_dimensions:
        df2 = df[df['维度'] == m]
        new_df = df2['情感类别'].value_counts()
        d = {}
        for key,values in zip(new_df.index,new_df.values):
            d[key] = values
        list_emotion.append([m,d])

    # 提取数据
    labels = [item[0] for item in list_emotion]
    positive_values = [item[1]['正面'] for item in list_emotion]
    negative_values = [item[1]['负面'] for item in list_emotion]

    # 计算百分比
    list2 = [
        {"value": int(pos), "percent": float(pos / (pos + neg))}
        for pos, neg in zip(positive_values, negative_values)
    ]

    list3 = [
        {"value": int(neg), "percent": float(neg / (pos + neg))}
        for pos, neg in zip(positive_values, negative_values)
    ]


    # 创建柱状图
    c = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add_xaxis(labels)
            .add_yaxis("正面", list2, stack="stack1", category_gap="50%")
            .add_yaxis("负面", list3, stack="stack1", category_gap="50%")
            .set_series_opts(
            label_opts=opts.LabelOpts(
                position="right",
                formatter=JsCode(
                    "function(x){return Number(x.data.percent * 100).toFixed(1) + '%';}"
                ),
            )
        )
        # .set_global_opts(
        #     yaxis_opts=opts.AxisOpts(max_=100)
        # )
        .render(f"./{name}/维度情感分类可视化.html")
    )


if __name__ == '__main__':
    list_name = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
    for name in list_name:
        lad_probability(name)