import pandas as pd
import seaborn as sns
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # 使用Agg后端
from scipy.stats import pearsonr


def main1(data,name):
    # 定义分析函数
    def calculate_pearson(df, group_cols, target_col=f'{name}'):
        results = []
        for (news_type, sector), group in df.groupby(group_cols):
            if len(group) < 2:  # 皮尔逊系数需要至少两个样本
                continue
            r, p = pearsonr(group['新闻强度'], group[target_col])
            results.append({
                '新闻类型': news_type,
                '板块': sector,
                '相关系数': r,
                'p值': p,
                '目标变量': target_col
            })
        return pd.DataFrame(results)

    # 分析不同新闻类型和板块对涨跌幅的影响
    results = calculate_pearson(data, ['新闻类型', '板块'], f'{name}')

    plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示
    # 可视化
    plt.figure(figsize=(15, 6))
    sns.barplot(data=results, x='板块', y='相关系数', hue='新闻类型')
    plt.xticks(rotation=45)
    plt.title(f'不同新闻类型和板块对股票的影响-{name}')
    plt.tight_layout()
    plt.savefig(f'./img/不同新闻类型和板块对股票的影响-{name}.png')
    plt.show()
    list_df = []
    new_df = data['板块'].value_counts()
    x_data = [x for x in new_df.index]
    for x in x_data:
        # 分析同一板块不同新闻强度的影响
        sector_df = data[data['板块'] == x]
        # 计算皮尔逊相关系数和p值
        r, p = pearsonr(sector_df['新闻强度'], sector_df[f'{name}'])
        corr_type = "正相关" if r > 0 else "负相关"

        new_data =pd.DataFrame()
        new_data['板块'] = [x]
        new_data['皮尔逊相关系数'] = [r]
        new_data['p值'] = [p]
        new_data['相关性'] = [corr_type]
        list_df.append(new_data)
        # 可视化优化
        plt.figure(figsize=(10, 6))
        scatter = sns.regplot(
            data=sector_df,
            x='新闻强度',
            y=f'{name}',
            scatter_kws={'s': 60, 'alpha': 0.7, 'color': '#2196F3'},
            line_kws={'color': '#FF5722', 'lw': 2}
        )

        # 添加统计信息标注
        plt.annotate(
            f'相关系数 r = {r:.2f} ({corr_type})\nP值 = {p:.3f}',
            xy=(1.12, 0.85),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round", alpha=0.1, facecolor="white")
        )

        # 添加标签和标题
        plt.xlabel('新闻强度（对数刻度）', fontsize=12)
        plt.ylabel(f'股票{name}', fontsize=12)
        plt.title(f'{x}：新闻强度与{name}的关系', fontsize=14, pad=20)

        # 优化刻度显示
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))  # 科学计数法显示x轴
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(f'./img/{x}：新闻强度与{name}的关系.png')
        plt.show()

    df1 = pd.concat(list_df,axis=0)

    # 保存结果到Excel
    with pd.ExcelWriter(f'./data/{name}-相关性分析数据.xlsx') as writer:
        results.to_excel(writer, sheet_name='不同新闻类型与不同版块（相关系数汇总）', index=False)
        df1.to_excel(writer, sheet_name='同一板块不同新闻强度（相关系数汇总）', index=False)


if __name__ == '__main__':
    df1 = pd.read_csv('国际新闻.csv')
    df2 = pd.read_csv('科技新闻.csv')
    df3 = pd.read_csv('娱乐新闻.csv')
    data = pd.concat([df1,df2,df3],axis=0)
    list_index = ['涨跌幅','涨跌额','收盘','成交量']
    for i in list_index:
        main1(data,i)


