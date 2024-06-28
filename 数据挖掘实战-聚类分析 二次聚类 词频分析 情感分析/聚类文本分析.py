import pandas as pd


def main(number):
    df = pd.read_csv('./聚类/聚类-{}/聚类结果.csv'.format(number))
    new_df = df['二级-聚类结果'].value_counts()
    x_data = [x for x in new_df.index]
    word2 = []
    sum_value = []
    for x in x_data:
        df1 = df[df['二级-聚类结果'] == x]
        d = {}
        for d1 in df1['fenci']:
            d1 = str(d1).split(' ')
            for d2 in d1:
                d[d2] = d.get(d2,0) + 1

        ls = list(d.items())
        ls.sort(key=lambda x:x[1],reverse=True)

        word1 = []
        sum_values = 0
        for key,values in ls:
            if values >= 10:
                word = f'{key}({values})'
                sum_values += values
                word1.append(word)
        word1 = ' '.join(word1)
        word2.append(word1)
        sum_value.append(sum_values)

    data = pd.DataFrame()
    data['高频词(频次)'] = word2
    data['二级-总频次'] = sum_value
    data['一级-总频次'] = sum(sum_value)

    return data


if __name__ == '__main__':
    data1 = main(0)
    data2 = main(1)
    data3 = main(2)

    data4 = pd.concat([data1,data2,data3],axis=0)

    sum_number = data4['二级-总频次'].sum()

    def pl1(x):
        x1 = int(x)
        pl = round(x1 / sum_number,4)
        pl = str(pl * 100) + '%'
        return pl

    data4['一级-频率(/%)'] = data4['一级-总频次'].apply(pl1)
    data4['二级-频率(/%)'] = data4['二级-总频次'].apply(pl1)

    data4.to_excel('感知形象表.xlsx',index=False)