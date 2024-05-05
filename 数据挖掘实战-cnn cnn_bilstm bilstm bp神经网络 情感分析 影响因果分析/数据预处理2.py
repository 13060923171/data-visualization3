import pandas as pd

df1 = pd.read_excel('data.xlsx')
df2 = pd.read_excel('评论特征.xlsx',sheet_name='生产设计')
df2_word = [word for word in df2['word']]
df3 = pd.read_excel('评论特征.xlsx',sheet_name='售前服务')
df3_word = [word for word in df3['word']]
df4 = pd.read_excel('评论特征.xlsx',sheet_name='个人体验')
df4_word = [word for word in df4['word']]


def demo1(x):
    x1 = str(x)
    x1_word = []
    for i in x1:
        if i in df2_word:
            x1_word.append(i)
    return len(x1_word)

def demo2(x):
    x1 = str(x)
    x1_word = []
    for i in x1:
        if i in df3_word:
            x1_word.append(i)
    return len(x1_word)

def demo3(x):
    x1 = str(x)
    x1_word = []
    for i in x1:
        if i in df4_word:
            x1_word.append(i)
    return len(x1_word)


df1['生产设计'] = df1['帖子正文'].apply(demo1)
df1['售前服务'] = df1['帖子正文'].apply(demo2)
df1['个人体验'] = df1['帖子正文'].apply(demo3)

df1.to_excel('特征词.xlsx')