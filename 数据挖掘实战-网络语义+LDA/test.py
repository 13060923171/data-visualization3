import pandas as pd

df = pd.read_excel('豆瓣评论.xlsx')


df1 = df.drop_duplicates(subset=['url'])
df2 = df.drop_duplicates(subset=['回复内容'])
df3 = df.drop_duplicates(subset=['发帖人'])
df4 = df.drop_duplicates(subset=['回复用户'])
print("帖子总数:",len(df1))
print("评论总数:",len(df2))
print("用户总数:",len(df3)+len(df4))



