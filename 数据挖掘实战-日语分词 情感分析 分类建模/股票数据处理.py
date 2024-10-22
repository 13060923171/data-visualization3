import pandas as pd

df = pd.read_excel('丰田股价20230531.xlsx')
print(df['日付'])
# 计算环比（当前值与前一个值的差异）
df['环比'] = df['終値'].pct_change()

# 判断环比结果，正数为1，负数或0为0
df['结果'] = df['环比'].apply(lambda x: 1 if x > 0 else 0)
df['日付'] = pd.to_datetime(df['日付'])
df.to_excel('股票.xlsx')