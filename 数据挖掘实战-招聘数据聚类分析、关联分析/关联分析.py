import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd


def demo(name):
    df = pd.read_csv('聚类结果.csv')
    df = df[df['聚类结果'] == name]
    def main1(x):
        x1 = str(x)
        x1 = x1.replace('1年以下', '经验不限').replace('一年以下', '经验不限')
        x1 = x1.replace('nan', '经验不限').replace('无经验', '经验不限')
        x1 = x1.replace('经验不限', '不限')
        return x1


    df['经验'] = df['经验'].apply(main1)


    def main2(x):
        x1 = str(x)
        x1 = x1.replace('统招本科', '本科').replace('中专/中技', '中专').replace('nan', '学历不限').replace('初中及以下', '学历不限').replace(
            'MBA/EMBA', '博士').replace('EMBA', '博士')
        x1 = x1.replace('中专', '大专以下').replace('高中', '大专以下')

        if '及以上' in x1:
            x1 = x1.split('及以上')
            x1 = x1[0]
        else:
            x1 = x1
        x1 = x1.replace('硕士', '硕士及以上')
        x1 = x1.replace('大专以下','学历不限').replace('博士','硕士及以上')
        return x1

    df['学历'] = df['学历'].apply(main2)


    def main3(x):
        x1 = str(x)
        if '面' in x1:
            return '面议'
        else:
            x2 = x1.split('-')
            x2 = x2[0]
            x2 = x2.replace('千','k').replace('万','0k')
            x2 = x2.replace('k','')
            if '.' in x2 and '0' in x2:
                x2 = float(x2) * 10
                return x2
            return x2

    def main4(x):
        try:
            if x == '面议':
                return '面议'
            else:
                x = float(x)
                if x <= 10:
                    return '月薪1万以下'
                elif 10 < x <= 20:
                    return '月薪1-2万'
                elif 20 < x <= 50:
                    return '月薪2-5万'
                elif x >= 50:
                    return '月薪5万以上'
        except:
            return '面议'



    df['月薪'] = df['月薪'].apply(main3)
    df['月薪类型'] = df['月薪'].apply(main4)

    data = pd.DataFrame()
    data['学历'] = df['学历']
    data['经验'] = df['经验']
    data['月薪'] = df['月薪类型']

    data1 = []
    for j,k,l in zip(data['学历'],data['经验'],data['月薪']):
        data1.append([j,k,l])


    te = TransactionEncoder()
    df_tf = te.fit_transform(data1)
    df = pd.DataFrame(df_tf,columns=te.columns_)
    frequent_itemsets = apriori(df,min_support=0.1,use_colnames= True)
    rules = association_rules(frequent_itemsets,metric = 'confidence',min_threshold = 0.15)
    rules = rules.drop(rules[rules.lift <1.0].index)
    rules.rename(columns = {'antecedents':'前项','consequents':'后项','support':'支持度 %','confidence':'置信度 %','lift':'提升'},inplace = True)

    data3 = pd.DataFrame(rules)
    data3 = data3.drop(['antecedent support','consequent support','leverage','conviction'],axis=1)

    def main5(x):
        x1 = str(x)
        x1 = x1.replace("frozenset(","").replace(")","")
        return x1

    data3['前项'] = data3['前项'].apply(main5)
    data3['后项'] = data3['后项'].apply(main5)
    data3.to_csv('./data/聚类{}-关联规则.csv'.format(name),encoding='utf-8-sig',index=False)


if __name__ == '__main__':
    df = pd.read_csv('聚类结果.csv')
    new_df = df['聚类结果'].value_counts()
    list1 = [x for x in new_df.index]
    for l in list1:
        demo(l)