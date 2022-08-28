import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv('new_推特.csv')
data['时间'] = pd.to_datetime(data['时间'],errors='ignore')
#将时间作为df的索引
data.Timestamp = pd.to_datetime(data['时间'],errors='ignore')
data.index = data.Timestamp


def year_number(x):
    str1 = str(x)
    if ("Guangdong" in str1 and "Hong" in str1) or ("Greater Bay Area" in str1) or ("Kong" in str1 and "Macao" in str1) or ("Guangdon" in str1 and "Macao" in str1):
        return 1
    else:
        return 0


def main1():
    data['年数据'] = data['推文'].apply(year_number)
    df_year = data['年数据'].resample('A-DEC').sum()
    df_year.sort_index(axis=0,ascending=False)

    x_data = []
    for x in df_year.index:
        x = str(x).split('-')[0]
        x_data.append(x)
    y_data = list(df_year.values)
    df = pd.DataFrame()
    df['年份'] = x_data
    df['数据'] = y_data
    df.to_csv('年数据.csv', encoding='utf-8-sig', index=False)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(20,9),dpi=500)
    plt.plot(x_data, y_data,linewidth=3,color='#b82410',label='年数据')
    plt.legend()
    plt.title('有关粤港澳大湾区-年推文数量变化趋势')
    plt.xlabel('年份')
    plt.ylabel('总数')
    plt.grid()
    plt.savefig('img01.png')
    plt.show()


def main2():
    data['年数据'] = data['推文'].apply(year_number)
    data1 = data[data['年数据'] == 1]
    df_year = data1['评论'].resample('A-DEC').sum()
    df_year.sort_index(axis=0,ascending=False)

    x_data = []
    for x in df_year.index:
        x = str(x).split('-')[0]
        x_data.append(x)
    y_data = list(df_year.values)
    df = pd.DataFrame()
    df['年份'] = x_data
    df['数据'] = y_data
    df.to_csv('关注数据.csv', encoding='utf-8-sig', index=False)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(20,9),dpi=500)
    plt.plot(x_data, y_data,linewidth=3,color='#58D68D',label='关注热度')
    plt.legend()
    plt.title('有关粤港澳大湾区-关注热度变化趋势')
    plt.xlabel('年份')
    plt.ylabel('总数')
    plt.grid()
    plt.savefig('img02.png')
    plt.show()


def main3():
    data1 = data[data['是否认证'] == '是']
    new_df = data1['发布者'].value_counts()
    new_df.to_csv('官媒数据.csv', encoding='utf-8-sig')
    x_data = list(new_df.index)[0:20]
    y_data = list(new_df.values)[0:20]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(20, 9),dpi=500)
    plt.style.use('ggplot')
    x_data.reverse()
    y_data.reverse()
    plt.barh(x_data, y_data)
    plt.title("top20官媒排名")
    plt.xlabel("数量")
    plt.savefig('img03.png')
    plt.show()


if __name__ == '__main__':
    main1()
    main2()
    main3()