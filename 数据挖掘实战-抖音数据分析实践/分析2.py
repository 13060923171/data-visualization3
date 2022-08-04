import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_excel('所有视频.xlsx')
new_df = df.groupby('keyword').agg('sum')
new_df.to_csv('数据总表.csv',encoding='utf-8-sig')
plt.rcParams['font.sans-serif'] = ['SimHei']
y_data = list(new_df.index)
x_data = list(new_df['share_count'])
x_data1 = list(new_df['comment_count'])
x_data2 = list(new_df['digg_count'])

plt.figure(figsize=(9,6))
plt.bar(y_data,x_data,color='#DAF7A6')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title("分享总数")
plt.xlabel("地区")
plt.ylabel("次数")
plt.savefig('分享总数.png')
plt.show()


plt.figure(figsize=(9,6))
plt.bar(y_data,x_data1,color='#A6E5F7')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title("评论总数")
plt.xlabel("地区")
plt.ylabel("次数")
plt.savefig('评论总数.png')
plt.show()


plt.figure(figsize=(9,6))
plt.bar(y_data,x_data2,color='#F7C2A6')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title("点赞总数")
plt.xlabel("地区")
plt.ylabel("次数")
plt.savefig('点赞总数.png')
plt.show()

# plt.xticks(rotation=65)
