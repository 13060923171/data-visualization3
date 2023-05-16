import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

df = pd.read_csv('聚类结果.csv')

def main1(x):
    x1 = str(x)
    x1 = x1.replace('在校/应届','不限').replace('1年以下','不限')
    return x1
df['经验'] = df['经验'].apply(main1)


def main2(x):
    x1 = int(x)
    if x1 <= 10000:
        return "平均薪资:1万以下"
    elif 10000 < x1 <= 20000:
        return "平均薪资:1万-2万区间"
    elif 20000 < x1 <= 30000:
        return "平均薪资:2万-3万区间"
    elif 30000 < x1 <= 50000:
        return "平均薪资:3万-5万区间"
    else:
        return "平均薪资:5万以上"

df['平均薪资'] = df['平均薪资'].apply(main2)


def main3(x):
    x1 = '聚类:' + str(x)
    return x1

df['聚类结果'] = df['聚类结果'].apply(main3)


data1 = []
for j,k,l,o,u,y,t in zip(df['搜索关键字'],df['学历'],df['经验'],df['职位'],df['公司规模'],df['平均薪资'],df['聚类结果']):
    data1.append([str(j),str(k),str(l),str(o),str(u),str(y),str(t)])


te = TransactionEncoder()
df_tf = te.fit_transform(data1)
df = pd.DataFrame(df_tf,columns=te.columns_)
#min_support=0.3表示一个项集在数据集中出现的频率
frequent_itemsets = apriori(df,min_support=0.3,use_colnames= True)
rules = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.15)
rules = rules.drop(rules[rules.lift <1.0].index)
rules.rename(columns={'antecedents':'前项','consequents':'后项','support':'支持度 %','confidence':'置信度 %','lift':'提升'},inplace = True)

data3 = pd.DataFrame(rules)
data3 = data3.drop(['antecedent support','consequent support','leverage','conviction'],axis=1)


def main5(x):
    x1 = str(x)
    x1 = x1.replace("frozenset(","").replace(")","")
    return x1


data3['前项'] = data3['前项'].apply(main5)
data3['后项'] = data3['后项'].apply(main5)

data3.sort_values(by=['置信度 %'],inplace=True,ascending=False)
data3.to_csv('关联规则.csv',encoding='utf-8-sig',index=False)

