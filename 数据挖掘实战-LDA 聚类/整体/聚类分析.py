import pandas as pd
import re
import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import random
from PIL import Image
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import plotly.express as px




def kmeans(df,weight,name):
    if not os.path.exists(f"./{name}/"):
        os.mkdir(f"./{name}/")

    # 计算特征相关系数矩阵
    correlation_matrix = weight.corr().round(2)

    # 生成交互式热力图
    fig = px.imshow(
        correlation_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=preprocessed_df.columns,  # 使用实际列名
        y=preprocessed_df.columns,
        color_continuous_scale="RdBu",  # 红蓝渐变色
        text_auto=True,  # 自动显示数值
        zmin=-1,  # 相关系数范围
        zmax=1,
        title="Feature Correlation Heatmap"
    )

    # 自定义布局
    fig.update_layout(
        width=1000,
        height=800,
        margin=dict(l=100, r=100, b=100, t=100),  # 避免长列名被截断
        xaxis=dict(tickangle=-45),  # 列名旋转45度
        title_font=dict(size=24)
    )

    # 保存HTML
    fig.write_html(
        f"./{name}/feature_correlation_heatmap.html",
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": False}
    )
    # 肘部法
    wcss = []
    silhouette_scores = []
    best_k = None
    # 设置聚类数量K的范围
    range_n_clusters = range(2, 11)
    # 计算每个K值对应的轮廓系数
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=111)
        labels = kmeans.fit_predict(weight)
        score = silhouette_score(weight, labels)
        silhouette_scores.append(score)
        kmeans.fit(weight)
        wcss.append(kmeans.inertia_)

        if best_k is None or score > silhouette_scores[best_k - 2]:
            best_k = n_clusters


    # Print the best K value and its corresponding silhouette score
    print(f"Best K value: {best_k}")
    print(f"Silhouette score for best K value: {silhouette_scores[best_k - 2]}")


    data = pd.DataFrame()
    data['聚类数量'] = range_n_clusters
    data['轮廓系数'] = silhouette_scores
    data['肘部系数'] = wcss
    data.to_csv(f'./{name}/轮廓系数.csv', encoding="utf-8-sig",index=False)
    # 绘制轮廓系数图
    plt.figure(figsize=(9,6),dpi=300)
    plt.plot(range_n_clusters, wcss, 'bo-', alpha=0.8)
    plt.xlabel('Number of clusters')
    plt.ylabel('wcss score')
    plt.title('wcss score for K-means clustering')
    plt.savefig(f'./{name}/肘部系数图.png')
    plt.show()

    plt.figure(figsize=(9,6),dpi=300)
    plt.plot(range_n_clusters, silhouette_scores, 'bo-', alpha=0.8)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for K-means clustering')
    plt.savefig(f'./{name}/轮廓系数图.png')
    plt.show()
    n_clusters = 2

    clf = KMeans(n_clusters=n_clusters, random_state=111)
    pre = clf.fit_predict(weight)

    df['聚类结果'] = pre
    df.to_csv(f'./{name}/聚类结果.csv', encoding="utf-8-sig",index=False)

    pca = PCA(n_components=2)  # 输出两维
    newData = pca.fit_transform(weight)  # 载入N维

    x = [n[0] for n in newData]
    y = [n[1] for n in newData]
    plt.figure(figsize=(9,6),dpi=300)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(x, y, c=pre, s=100)
    plt.title("聚类图")
    plt.savefig(f'./{name}/聚类图.jpg')
    plt.show()

    if not os.path.exists(f"./{name}/image/"):
        os.mkdir(f"./{name}/image/")
    new_df = df['聚类结果'].value_counts()
    x_data = [x for x in new_df.index]
    for x in x_data:
        df1 = df[df['聚类结果'] == x]
        list_text = []
        for t in df1['words']:
            # 把数据分开
            t = str(t).split(" ")
            for i in t:
                list_text.append(i)

        def color_func(word, font_size, position, orientation, random_state=None,
                       **kwargs):
            return "hsl({}, 100%, 50%)".format(np.random.randint(0, 300))

            # 读取背景图片
        background_Image = np.array(Image.open('images.png'))
        text = ' '.join(list_text)
        wc = WordCloud(
            collocations=False,  # 禁用词组
            font_path='simhei.ttf',  # 中文字体路径
            margin=3,  # 词云图边缘宽度
            mask=background_Image,  # 背景图形
            scale=3,  # 放大倍数
            max_words=200,  # 最多词个数
            random_state=42,  # 随机状态
            width=1600,  # 提高分辨率
            height=1200,
            min_font_size=10,  # 调大最小字体
            max_font_size=80,  # 调大最大字体
            background_color='white',  # 背景颜色
            color_func=color_func  # 字体颜色函数
        )
        # 生成词云
        # 直接从词频生成词云
        wc.generate_from_text(text)
        # 保存高清图片
        wc.to_file(f'./{name}/image/聚类{x}-词云图.png')

if __name__ == '__main__':
    # 加载数据（替换为你的数据路径）
    data = pd.read_excel("整体教师用户画像.xlsx")

    # 1. 列名清洗
    data.columns = [col.strip() for col in data.columns]
    numeric_cols = ['知识分享方向',"知识分享主题数量", "总发帖量", "平均发帖长度", "平均喜好", "平均评论数", "平均收藏", "粉丝数量", "关注数量","发帖类型", "IP属地", "博主类型", "性别"]

    data1 = data[numeric_cols].copy()
    label_encoder = LabelEncoder()
    data1['发帖类型'] = label_encoder.fit_transform(data1['发帖类型'])
    data1['IP属地'] = label_encoder.fit_transform(data1['IP属地'])
    data1['博主类型'] = label_encoder.fit_transform(data1['博主类型'])
    data1['性别'] = label_encoder.fit_transform(data1['性别'])
    # 5. 特征缩放
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data1)
    preprocessed_df = pd.DataFrame(scaled_data, columns=data1.columns)

    kmeans(data,preprocessed_df,'整体教师')

