import pandas as pd
import re
import numpy as np
df1 = pd.read_excel('猎聘-xlsx.xlsx')
df2 = pd.read_excel('智联-xlsx.xlsx')


def demo1():
    def main1(x):
        x1 = str(x).split('\n\n')
        for i in x1:
            if '要求' in i and '学历' in i and ('大专' in i or '本科' in i or "研究生" in i or "博士" in i):
                i = i.replace('_x000D_','').strip(' ')
                if '职责' in i:
                    i = str(i).split('。')
                    for j in i:
                        if '要求' in j and '学历' in j and ('大专' in j or '本科' in j or "研究生" in j or "博士" in j):
                            return j
                else:
                    return i

    df1['全文'] = df1['全文'].apply(main1)

    def main2(x):
        x1 = str(x)
        x2 = re.compile("(.*?)相关专业")
        x3 = x2.findall(x1)
        if len(x3) != 0:
            return x3[0]
        else:
            x2 = re.compile("(.*?)专业")
            x3 = x2.findall(x1)
            if len(x3) != 0:
                return x3[0]
            else:
                return '不限'

    df1['专业'] = df1['全文'].apply(main2)
    data1 = pd.DataFrame()
    data1['岗位名称'] = df1['标题']
    data1['学历'] = df1['学历要求']
    data1['月薪'] = df1['薪资']
    data1['经验'] = df1['经验要求']
    data1['专业'] = df1['专业']
    data1['专业知识'] = df1['全文']

    def main3(x):
        x1 = str(x)
        x1 = x1.replace("\n",'').replace(" ","")
        if x1 == "None":
            return np.NAN
        else:
            return x1
    data1['专业知识'] = data1['专业知识'].apply(main3)
    data1 = data1.dropna(subset=['专业知识'],axis=0)
    return data1


def demo2():
    def main2(x):
        x1 = str(x)
        x2 = re.compile("、(.*?)相关专业")
        x3 = x2.findall(x1)
        if len(x3) != 0:
            return x3[0]
        else:
            x2 = re.compile("、(.*?)专业")
            x3 = x2.findall(x1)
            if len(x3) != 0:
                return x3[0]
            else:
                return '不限'

    df2['专业'] = df2['岗位描述'].apply(main2)
    data1 = pd.DataFrame()
    data1['岗位名称'] = df2['职位昵称']
    data1['学历'] = df2['学历']
    data1['月薪'] = df2['薪资']
    data1['经验'] = df2['经验']
    data1['专业'] = df2['专业']
    data1['专业知识'] = df2['岗位描述']
    def main3(x):
        x1 = str(x)
        x1 = x1.replace("\n", '').replace(" ", "")
        if x1 == "nan":
            return np.NAN
        else:
            return x1
    data1['专业知识'] = data1['专业知识'].apply(main3)
    data1 = data1.dropna(subset=['专业知识'], axis=0)
    return data1


data1 = demo1()
data2 = demo2()
data3 = pd.concat([data1,data2],axis=0)
data3.to_csv('data.csv',encoding='utf-8-sig',index=False)