import pandas as pd

df = pd.read_excel('demo.xlsx')
#填充
df['出发时间'] = df['出发时间'].fillna(method='ffill')

#时间处理
def main1(x):
    x1 = str(x).split('/')
    return x1[-1]

#时间筛选
df['时间'] = df['出发时间'].apply(main1)
df['时间'] = pd.to_datetime(df['时间'])
df.index = df['时间']
new_df = df['2020-01-01':'2023-12-31']
#保存数据
new_df.to_csv('new_data.csv',encoding='utf-8-sig',index=False)