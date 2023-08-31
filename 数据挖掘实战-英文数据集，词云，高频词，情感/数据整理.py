import pandas as pd

# df1 = pd.read_excel('google处理结果（已清洗）.xlsx')
# df2 = pd.read_excel('meta检索结果.xlsx')
#
# df1['来源'] = 'google new'
# df2['来源'] = 'meta'
# df1['文本内容'] = df1['正文']
# df2['文本内容'] = df2['评论时间']
# df2['时间'] = df2['评论正文']
# df1 = df1[['文本内容','来源','时间']]
# df2 = df2[['文本内容','来源','时间']]
# df3 = pd.concat([df1,df2],axis=0)
# df3.to_excel('媒体数据.xlsx',encoding='utf-8-sig',index=False)

df1 = pd.read_excel('imdb检索结果.xlsx')
df2 = pd.read_excel('youtube搜索结果（已清洗）.xlsx')
df3 = pd.read_excel('烂番茄检索结果.xlsx')
df4 = pd.read_excel('twitter检索结果.xlsx')

df1['来源'] = 'imdb'
df1['文本内容'] = df1['content']
df1['时间'] = df1['date']
df2['来源'] = 'youtube'
df2['文本内容'] = df2['cms']
df2['时间'] = df2['publishedAt']
df3['来源'] = '烂番茄'
df3['文本内容'] = df3['fullcontent']
df3['时间'] = df3['rate']
df4 = df4.dropna(subset=['发文时间'],axis=0)
df4.index = pd.to_datetime(df4['发文时间'])
df4 = df4['2019-07-26':'2020-01-26']
df4['来源'] = 'Twitter'
df4['文本内容'] = df4['全文']
df4['时间'] = df4['发文时间']
df1 = df1[['文本内容','来源','时间']]
df2 = df2[['文本内容','来源','时间']]
df3 = df3[['文本内容','来源','时间']]
df4 = df4[['文本内容','来源','时间']]
df5 = pd.concat([df1,df2,df3,df4],axis=0)
df5.to_excel('评论数据.xlsx',encoding='utf-8-sig',index=False)
df5 = df5[['文本内容']]
df5.to_excel('数据.xlsx',encoding='utf-8-sig',index=False)
