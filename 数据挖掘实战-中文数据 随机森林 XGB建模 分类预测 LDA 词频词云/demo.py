import pandas as pd

data1 = pd.read_excel('./xgb/new_xgb_data.xlsx')
data1 = data1.drop_duplicates(subset=['content'])
data2 = pd.read_csv('new_data.csv')
# 选择需要保留的列
data4 = data2[['content', 'lat', 'lng']]
# 内连接合并数据
data3 = pd.merge(data1, data4, on='content', how='inner')
# 保存结果
data3.to_csv('./xgb/new_xgb_data.csv', index=False,encoding='utf-8-sig')
