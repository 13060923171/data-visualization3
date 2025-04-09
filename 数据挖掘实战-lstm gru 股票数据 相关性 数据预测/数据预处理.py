import pandas as pd
from io import StringIO

df = pd.read_csv('merged_output.csv')
df['日期'] = pd.to_datetime(df['日期'])


def main1(name):
    # 按关键词、日期（升序）、板块排序
    df_sorted = df.sort_values(by=['关键词', '日期','资讯数据','新闻类型','新闻强度','板块'])

    # 将指标扩展为列
    df_pivot = df_sorted.pivot_table(
        index=['关键词', '日期','资讯数据','新闻类型','新闻强度','板块'],
        columns='指标',
        values='值',
        aggfunc='first'
    ).reset_index()

    # 重命名列并整理格式
    df_pivot.columns.name = None
    df_pivot = df_pivot.rename_axis(None, axis=1)

    new_df = df_pivot[df_pivot['新闻类型'] == name]
    new_df.to_csv(f'{name}.csv',encoding='utf-8-sig',index=False)


list_name = ['科技新闻','国际新闻','娱乐新闻']
for n in list_name:
    main1(n)

