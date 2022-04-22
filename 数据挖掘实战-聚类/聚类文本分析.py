import pandas as pd

df = pd.read_csv('聚类分类.csv')

def main1():
    df1 = df[df['聚类结果'] == 0]
    zb = len(df1) / len(df)
    zb_bfb = '%0.2lf' %(zb * 100) + '%'
    counts = {}
    for w in df1['word']:
        w = w.split(' ')
        for i in w:
            counts[i] = counts.get(i, 0) + 1
    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    for key,values in ls[0:20]:
        x_data.append(key)
    return zb_bfb,x_data

def main2():
    df1 = df[df['聚类结果'] == 1]
    zb = len(df1) / len(df)
    zb_bfb = '%0.2lf' %(zb * 100) + '%'
    counts = {}
    for w in df1['word']:
        try:
            w = w.split(' ')
        except:
            w = [w]
        for i in w:
            counts[i] = counts.get(i, 0) + 1
    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    for key,values in ls[0:20]:
        x_data.append(key)
    return zb_bfb,x_data

def main3():
    df1 = df[df['聚类结果'] == 2]
    zb = len(df1) / len(df)
    zb_bfb = '%0.2lf' %(zb * 100) + '%'
    counts = {}
    for w in df1['word']:
        w = w.split(' ')
        for i in w:
            counts[i] = counts.get(i, 0) + 1
    ls = list(counts.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    x_data = []
    for key,values in ls[0:20]:
        x_data.append(key)
    return zb_bfb,x_data

if __name__ == '__main__':
    result1,result2 = main1()
    result3, result4 = main2()
    result5, result6 = main3()

    data = pd.DataFrame()
    data['第0聚类top关键词'] = [result2]
    data['第0聚类占比'] = result1
    data['第1聚类top关键词'] = [result4]
    data['第1聚类占比'] = result3
    data['第2聚类top关键词'] = [result6]
    data['第2聚类占比'] = result5

    data.to_csv('聚类结果分析.csv',encoding='utf-8-sig')