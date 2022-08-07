import pandas as pd
import re
df = pd.read_excel('data.xlsx')
df = df.dropna(subset=['fulltext'], axis=0)
content = df['fulltext'].drop_duplicates(keep='first')
content = content.dropna(how='any')

list_zz = []
list_zg = []
list_ms = []
for c in content:
    # c = str(c).replace('\r\n','').strip(' ').split('：')
    # with open('任职要求')
    zz = re.compile('.*?职责:(.*?):')
    zz1 = zz.findall(c)
    zg = re.compile('.*?要求:(.*?):')
    zg1 = zg.findall(c)
    ms = re.compile('.*?描述:(.*?):')
    ms1 = ms.findall(c)
    if len(zz1) !=0:
        list_zz.append(zz1[0])
    if len(zg1) !=0:
        list_zg.append(zg1[0])
    if len(ms1) !=0:
        list_ms.append(ms1[0])


df = pd.DataFrame()
df['岗位职责'] = list_zz
df.to_csv('岗位职责.csv',encoding='utf-8-sig',index=False)

df1 = pd.DataFrame()
df1['任职要求'] = list_zg
df1.to_csv('任职要求.csv',encoding='utf-8-sig',index=False)

df2 = pd.DataFrame()
df2['岗位描述'] = list_ms
df2.to_csv('岗位描述.csv',encoding='utf-8-sig',index=False)

