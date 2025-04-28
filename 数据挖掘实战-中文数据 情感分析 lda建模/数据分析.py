import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')
sns.set_style(style="whitegrid")

import matplotlib.dates as mdates


df1 = pd.read_excel('博文表.xlsx')

def data_time(x):
    x1 = str(x).split(" ")
    return x1[0]


df1['发布时间'] = df1['发布时间'].apply(data_time)
df1['发布时间'] = pd.to_datetime(df1['发布时间'],format='%Y年%m月%d日')

df2 = pd.read_excel('评论表.xlsx')

df2['评论时间'] = df2['评论时间'].apply(data_time)
df2['评论时间'] = pd.to_datetime(df2['评论时间'])

def main1(df1,df2):
    list_x1 = ['2023-08-24','2023-08-25','2023-10-01','2024-03-01']
    list_y1 = []
    list_y2 = []

    phase1_y1 = df1[(df1['发布时间'] == '2023-08-24')]
    list_y1.append(len(phase1_y1))
    phase2_y1 = df1[(df1['发布时间'] >= '2023-08-25') & (df1['发布时间'] <= '2023-09-30')]
    list_y1.append(len(phase2_y1))
    phase3_y1 = df1[(df1['发布时间'] >= '2023-10-01') & (df1['发布时间'] <= '2024-02-28')]
    list_y1.append(len(phase3_y1))
    phase4_y1 = df1[(df1['发布时间'] >= '2024-03-01') & (df1['发布时间'] <= '2024-07-31')]
    list_y1.append(len(phase4_y1))

    phase1_y2 = df2[(df2['评论时间'] == '2023-08-24')]
    list_y2.append(len(phase1_y2))
    phase2_y2 = df2[(df2['评论时间'] >= '2023-08-25') & (df2['评论时间'] <= '2023-09-30')]
    list_y2.append(len(phase2_y2))
    phase3_y2 = df2[(df2['评论时间'] >= '2023-10-01') & (df2['评论时间'] <= '2024-02-28')]
    list_y2.append(len(phase3_y2))
    phase4_y2 = df2[(df2['评论时间'] >= '2024-03-01') & (df2['评论时间'] <= '2024-07-31')]
    list_y2.append(len(phase4_y2))

    # 1. 设置全局样式（解决中文显示问题 + 美化）
    plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体（根据系统安装的字体调整）
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
    plt.rcParams['font.size'] = 12  # 全局字体大小

    # 2. 创建画布与坐标轴
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)  # 设置画布尺寸和分辨率

    # 3. 绘制折线图（带数据标记）
    line1, = ax.plot(
        list_x1,
        list_y1,
        color='#2B7DDE',  # 蓝色系
        marker='o',  # 圆形标记
        markersize=8,  # 标记尺寸
        linestyle='--',  # 虚线样式
        linewidth=2,  # 线宽
        label='博文'  # 图例标签
    )

    line2, = ax.plot(
        list_x1,
        list_y2,
        color='#FF6B6B',  # 红色系
        marker='s',  # 方形标记
        markersize=8,
        linestyle='-.',  # 点划线
        linewidth=2,
        label='评论'
    )

    # 4. 添加图表元素
    ax.set_title('舆情发展阶段数据趋势分析', fontsize=16, pad=20)  # 标题
    ax.set_xlabel('发展阶段', fontsize=14, labelpad=10)  # X轴标签
    ax.set_ylabel('数据量', fontsize=14, labelpad=10)  # Y轴标签

    # 5. 定制刻度样式
    ax.tick_params(axis='both', which='major', labelsize=12)  # 刻度标签字号
    plt.xticks(rotation=45, ha='right')  # X轴标签旋转45度，右对齐

    # 6. 添加辅助元素
    ax.grid(True, alpha=0.3)  # 半透明网格线
    ax.legend(frameon=True, shadow=True, fontsize=12)  # 带阴影的图例

    # 7. 数据标签标注（在标记点上方显示数值）
    for x, y1, y2 in zip(list_x1, list_y1, list_y2):
        ax.text(x, y1 + 5, f'{y1}', ha='center', va='bottom', fontsize=10, color=line1.get_color())
        ax.text(x, y2 + 5, f'{y2}', ha='center', va='bottom', fontsize=10, color=line2.get_color())

    # 8. 调整布局
    plt.tight_layout()  # 自动适配元素间距

    # 9. 保存或显示
    plt.savefig('舆情发展阶段数据趋势分析.png', bbox_inches='tight', dpi=300)  # 保存高清图
    plt.show()


def monthly_analysis(df1, df2):
    # 按月统计博文数量
    df_posts = df1.set_index('发布时间').resample('M').size().reset_index(name='博文数量')
    df_posts['月份'] = df_posts['发布时间'].dt.strftime('%Y-%m')

    # 按月统计评论数量
    df_comments = df2.set_index('评论时间').resample('M').size().reset_index(name='评论数量')
    df_comments['月份'] = df_comments['评论时间'].dt.strftime('%Y-%m')

    # 合并数据（处理可能存在的空月份）
    all_months = pd.date_range(start=min('2023-08', '2023-08'),
                               end=max('2024-08', '2024-08'),
                               freq='MS').strftime('%Y-%m').tolist()

    final_df = pd.DataFrame({'月份': all_months})
    final_df = final_df.merge(df_posts[['月份', '博文数量']], on='月份', how='left')
    final_df = final_df.merge(df_comments[['月份', '评论数量']], on='月份', how='left')
    final_df.fillna(0, inplace=True)
    # 转换为datetime类型用于绘图
    final_df['日期'] = pd.to_datetime(final_df['月份'])

    # 设置绘图样式
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

    # 绘制双折线图
    ax.plot(final_df['日期'],
            final_df['博文数量'],
            marker='o',
            color='#1f77b4',
            linewidth=2,
            label='博文数量')

    ax.plot(final_df['日期'],
            final_df['评论数量'],
            marker='s',
            color='#ff7f0e',
            linewidth=2,
            linestyle='--',
            label='评论数量')

    # 设置图表元素
    ax.set_title('月度舆情数据趋势分析', fontsize=16, pad=20)
    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('数量', fontsize=12)

    # 设置x轴格式
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')

    # 添加数据标签
    for x, y in zip(final_df['日期'], final_df['博文数量']):
        ax.text(x, y + 0.5, f'{int(y)}', ha='center', va='bottom', fontsize=8)

    for x, y in zip(final_df['日期'], final_df['评论数量']):
        ax.text(x, y + 0.5, f'{int(y)}', ha='center', va='bottom', fontsize=8)

    # 美化设置
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, shadow=True)
    plt.tight_layout()

    # 保存并显示
    plt.savefig('月度舆情分析.png', bbox_inches='tight', dpi=300)
    plt.show()




def data_emotion(df):
    new_df = df['label'].value_counts()
    d = {}
    for x,y in zip(new_df.index,new_df.values):
        d[x] =y
    return d

def main2(df1,df2):
    list_x1 = ['2023-08-24','2023-08-25','2023-10-01','2024-03-01']
    list_y1 = []
    list_y2 = []

    phase1_y1 = df1[(df1['发布时间'] == '2023-08-24')]
    y1_d1 = data_emotion(phase1_y1)
    list_y1.append(y1_d1)
    phase2_y1 = df1[(df1['发布时间'] >= '2023-08-25') & (df1['发布时间'] <= '2023-09-30')]
    y1_d2 = data_emotion(phase2_y1)
    list_y1.append(y1_d2)
    phase3_y1 = df1[(df1['发布时间'] >= '2023-10-01') & (df1['发布时间'] <= '2024-02-28')]
    y1_d3 = data_emotion(phase3_y1)
    list_y1.append(y1_d3)
    phase4_y1 = df1[(df1['发布时间'] >= '2024-03-01') & (df1['发布时间'] <= '2024-07-31')]
    y1_d4 = data_emotion(phase4_y1)
    list_y1.append(y1_d4)

    phase1_y2 = df2[(df2['评论时间'] == '2023-08-24')]
    y2_d1 = data_emotion(phase1_y2)
    list_y2.append(y2_d1)
    phase2_y2 = df2[(df2['评论时间'] >= '2023-08-25') & (df2['评论时间'] <= '2023-09-30')]
    y2_d2 = data_emotion(phase2_y2)
    list_y2.append(y2_d2)
    phase3_y2 = df2[(df2['评论时间'] >= '2023-10-01') & (df2['评论时间'] <= '2024-02-28')]
    y2_d3 = data_emotion(phase3_y2)
    list_y2.append(y2_d3)
    phase4_y2 = df2[(df2['评论时间'] >= '2024-03-01') & (df2['评论时间'] <= '2024-07-31')]
    y2_d4 = data_emotion(phase4_y2)
    list_y2.append(y2_d4)

    # 定义转换函数（处理单条数据）
    def convert_to_percentage(data_list):
        percentages = []
        for phase in data_list:
            total = sum(phase.values())
            percent = {k: round(v / total * 100, 1) for k, v in phase.items()}
            percentages.append(percent)
        return percentages

    # 转换两个数据集
    y1_percent = convert_to_percentage(list_y1)
    y2_percent = convert_to_percentage(list_y2)

    # 提取各情感占比（按阶段排序）
    emotions = ['正面', '负面', '中立']
    y1_values = [[phase[e] for e in emotions] for phase in y1_percent]
    y2_values = [[phase[e] for e in emotions] for phase in y2_percent]

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100)

    # 2. 定义颜色方案
    colors = ['#66C2A5', '#FC8D62', '#8DA0CB']  # 绿/橙/蓝

    # 3. 绘制数据序列1（示例：博文情感）
    bottom = np.zeros(4)
    for idx, emotion in enumerate(emotions):
        values = [phase[idx] for phase in y1_values]
        ax1.bar(list_x1, values, bottom=bottom, label=emotion, color=colors[idx], edgecolor='white')

        # 添加数据标签
        for i, (val, b) in enumerate(zip(values, bottom)):
            ax1.text(i, b + val / 2, f'{val}%', ha='center', va='center', color='white', fontsize=10)
        bottom += values

    ax1.set_title('博文情感阶段占比分析', fontsize=14, pad=20)
    ax1.set_ylabel('百分比 (%)')
    ax1.grid(axis='y', alpha=0.3)

    # 4. 绘制数据序列2（示例：评论情感）
    bottom = np.zeros(4)
    for idx, emotion in enumerate(emotions):
        values = [phase[idx] for phase in y2_values]
        ax2.bar(list_x1, values, bottom=bottom, label=emotion, color=colors[idx], edgecolor='white')

        # 添加数据标签
        for i, (val, b) in enumerate(zip(values, bottom)):
            ax2.text(i, b + val / 2, f'{val}%', ha='center', va='center', color='white', fontsize=10)
        bottom += values

    ax2.set_title('评论情感阶段占比分析', fontsize=14, pad=20)
    ax2.set_ylabel('百分比 (%)')
    ax2.grid(axis='y', alpha=0.3)

    # 5. 统一设置
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper right', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig('舆情发展阶段情感分析.png', bbox_inches='tight', dpi=300)  # 保存高清图
    plt.show()


def main3(df1,df2):
    df3 = pd.DataFrame()
    df3['时间'] = df1['发布时间']
    df3['情感词'] = df1['情感词']

    df4 = pd.DataFrame()
    df4['时间'] = df2['评论时间']
    df4['情感词'] = df2['情感词']

    df5 = pd.concat([df3, df4], axis=0)

    list_x1 = ['酝酿期', '爆发期', '蔓延期', '衰退期']
    list_y2 = []

    phase1_y2 = df5[(df5['时间'] == '2023-08-24')]
    list_y2.append(phase1_y2)
    phase2_y2 = df5[(df5['时间'] >= '2023-08-25') & (df5['时间'] <= '2023-09-30')]
    list_y2.append(phase2_y2)
    phase3_y2 = df5[(df5['时间'] >= '2023-10-01') & (df5['时间'] <= '2024-02-28')]
    list_y2.append(phase3_y2)
    phase4_y2 = df5[(df5['时间'] >= '2024-03-01') & (df5['时间'] <= '2024-07-31')]
    list_y2.append(phase4_y2)

    def word_process(df,name):
        df1 = df.dropna(subset=['情感词'],axis=0)
        d = {}
        list_text = []
        for t in df1['情感词']:
            # 把数据分开
            t = str(t).split(" ")
            for i in t:
                list_text.append(i)
                d[i] = d.get(i, 0) + 1

        ls = list(d.items())
        ls.sort(key=lambda x: x[1], reverse=True)
        x_data = []
        y_data = []
        for key, values in ls[:10]:
            x_data.append(key)
            y_data.append(values)

        data = pd.DataFrame()
        data['word'] = x_data
        data['counts'] = y_data

        data.to_excel(f'{name}-情感词Top10.xlsx', index=False)

    for x,y in zip(list_x1,list_y2):
        word_process(y,x)

if __name__ == '__main__':
    # main1(df1,df2)
    # main2(df1, df2)
    # main3(df1, df2)

    # 调用函数
    monthly_analysis(df1, df2)