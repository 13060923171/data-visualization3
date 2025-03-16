import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

import pandas as pd
import numpy as np
import os


def post_type(df,name):
    # 示例数据
    new_df = df['发帖类型'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#dad7cd','#a3b18a']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"发帖类型分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/发帖类型分布情况.png')


def kmean_type(df,name):
    # 示例数据
    new_df = df['聚类结果'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#f9dbbd','#ffa5ab','#dad7cd','#a3b18a','#3a7ca5','#d9dcd6']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    # 添加标题
    plt.title(f"聚类结果分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'聚类结果分布情况.png')


def lda_type1(df,name):
    # 示例数据
    new_df = df['知识分享方向'].value_counts()
    main_categories = [f"主题-{x+1}" for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#0fa3b1','#b5e2fa','#f7a072']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"知识分享方向分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/知识分享方向分布情况.png')

def lda_type2(df,name):
    # 示例数据
    new_df = df['知识分享主题数量'].value_counts()
    main_categories = [f"主题数量:{x}" for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#0fa3b1','#b5e2fa','#f7a072']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"知识分享主题数量分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/知识分享主题数量分布情况.png')

def blogger_type(df,name):
    # 示例数据
    new_df = df['博主类型'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#d8e2dc','#e6ccb2','#ffcad4','#f4acb7','#9d8189','#f4d35e']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"博主类型分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/博主类型分布情况.png')

def six_type(df,name):
    # 示例数据
    new_df = df['性别'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#17c3b2','#fe6d73']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"性别分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/性别分布情况.png')

def ip_type(df,name):
    # 示例数据
    new_df = df['IP属地'].value_counts()
    categories = [x for x in new_df.index]
    values = [y for y in new_df.values]

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(16, 8))
    # 设置全局字体
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制柱状图
    bars = ax.bar(categories,
                  values,
                  color='#edafb8',
                  edgecolor='black', linewidth=1)

    # 自定义样式
    ax.set_title(f'IP属地分布情况', fontsize=18, pad=20)
    ax.set_xlabel('IP归属', fontsize=14)
    ax.set_ylabel('人数', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱子上方显示数值
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移量
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12,
                        color='black')

    add_labels(bars)

    # 调整刻度标签
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # 自动调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/IP属地分布情况.png')

def post_number(x):
    x1 = int(x)
    if x1 <= 21:
        return '发帖频率：一般'
    elif 21 < x1 <= 28:
        return '发帖频率：较为积极'
    elif 28 < x1 <= 77:
        return '发帖频率：积极'
    else:
        return '发帖频率：频繁'


def post_type(df,name):
    # 示例数据
    new_df = df['总发帖量'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#84dcc6', '#a5ffd6','#ffa69e', '#ff686b']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"总发帖量分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/总发帖量分布情况.png')



def post_length(x):
    x1 = int(x)
    if x1 <= 75:
        return '发帖综合长度：低'
    elif 75 < x1 <= 145:
        return '发帖综合长度：中'
    elif 145 < x1 <= 266:
        return '发帖综合长度：高'
    else:
        return '发帖综合长度：超高'


def length_type(df,name):
    # 示例数据
    new_df = df['平均发帖长度'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#84dcc6', '#a5ffd6','#ffa69e', '#ff686b']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"平均发帖长度分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/平均发帖长度分布情况.png')

def mean_like(x):
    x1 = int(x)
    if x1 <= 34:
        return '点赞人数综合度：低'
    elif 34 < x1 <= 109:
        return '点赞人数综合度：中'
    elif 109 < x1 <= 264:
        return '点赞人数综合度：偏高'
    else:
        return '点赞人数综合度：高'

def like_type(df,name):
    # 示例数据
    new_df = df['平均喜好'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#84dcc6', '#a5ffd6','#ffa69e', '#ff686b']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"平均喜好分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/平均喜好分布情况.png')

def mean_comment(x):
    x1 = int(x)
    if x1 <= 3:
        return '评论综合人数：低'
    elif 3 < x1 <= 7:
        return '评论综合人数：中'
    elif 7 < x1 <= 20:
        return '评论综合人数：偏高'
    else:
        return '评论综合人数：高'

def comment_type(df,name):
    # 示例数据
    new_df = df['平均评论数'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#84dcc6', '#a5ffd6','#ffa69e', '#ff686b']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"平均评论数分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/平均评论数分布情况.png')

def mean_collect(x):
    x1 = int(x)
    if x1 <= 23:
        return '收藏综合人数：低'
    elif 23 < x1 <= 77:
        return '收藏综合人数：中'
    elif 77 < x1 <= 203:
        return '收藏综合人数：偏高'
    else:
        return '收藏综合人数：高'

def collect_type(df,name):
    # 示例数据
    new_df = df['平均收藏'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#84dcc6', '#a5ffd6','#ffa69e', '#ff686b']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"平均收藏分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/平均收藏分布情况.png')

def fan_number(x):
    x1 = int(x)
    if x1 <= 580:
        return '粉丝数量：低'
    elif 580 < x1 <= 2809:
        return '粉丝数量：中'
    elif 2809 < x1 <= 9856:
        return '粉丝数量：偏高'
    else:
        return '粉丝数量：高'

def fan_type(df,name):
    # 示例数据
    new_df = df['粉丝数量'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#84dcc6', '#a5ffd6','#ffa69e', '#ff686b']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"粉丝数量分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/粉丝数量分布情况.png')

def focus_number(x):
    x1 = int(x)
    if x1 <= 21:
        return '关注数量：低'
    elif 21 < x1 <= 67:
        return '关注数量：中'
    elif 67 < x1 <= 267:
        return '关注数量：偏高'
    else:
        return '关注数量：高'

def focus_type(df,name):
    # 示例数据
    new_df = df['关注数量'].value_counts()
    main_categories = [x for x in new_df.index]
    main_sizes = [y for y in new_df.values]

    # 颜色配置
    colors_main = ['#84dcc6', '#a5ffd6','#ffa69e', '#ff686b']

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = 'SimHei'

    # 绘制大饼图（主分类）
    wedges_main, texts_main, autotexts_main = ax.pie(
        main_sizes,
        labels=main_categories,
        colors=colors_main,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(main_sizes) / 100)})',
        startangle=40,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    # 调整主饼图文字样式
    plt.setp(autotexts_main, size=10, weight="bold", color='black')
    plt.setp(texts_main, size=12)

    # 添加中心空白区域（可选）
    centre_circle = plt.Circle((0, 0), 0.2, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例和标题
    ax.legend(
        wedges_main,
        main_categories,
        loc="upper left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    if not os.path.exists(f"./聚类-{name}/"):
        os.mkdir(f"./聚类-{name}/")
    # 添加标题
    plt.title(f"关注数量分布情况", y=1.05, fontsize=16)
    # 调整布局
    plt.tight_layout()
    plt.savefig(f'./聚类-{name}/关注数量分布情况.png')

if __name__ == '__main__':
    df = pd.read_csv('./整体教师/聚类结果.csv')
    kmean_type(df,"整体")
    df['总发帖量'] = df['总发帖量'].apply(post_number)
    df['平均发帖长度'] = df['平均发帖长度'].apply(post_length)
    df['平均喜好'] = df['平均喜好'].apply(mean_like)
    df['平均评论数'] = df['平均评论数'].apply(mean_comment)
    df['平均收藏'] = df['平均收藏'].apply(mean_collect)
    df['粉丝数量'] = df['粉丝数量'].apply(fan_number)
    df['关注数量'] = df['关注数量'].apply(focus_number)
    df1 = df['聚类结果'].value_counts()
    for d in df1.index:
        df2 = df[df['聚类结果'] == d]
        post_type(df2,d)
        lda_type1(df2,d)
        lda_type2(df2,d)
        blogger_type(df2,d)
        six_type(df2,d)
        ip_type(df2,d)
        post_type(df2,d)
        length_type(df2,d)
        like_type(df2,d)
        comment_type(df2,d)
        collect_type(df2,d)
        fan_type(df2,d)
        focus_type(df2,d)