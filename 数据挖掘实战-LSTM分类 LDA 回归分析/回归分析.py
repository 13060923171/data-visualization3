import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df1 = pd.read_excel('train.xlsx')
df2 = pd.read_excel('new_test.xlsx')
df3 = pd.concat([df1, df2], axis=0)

def process(x):
    x1 = str(x).replace("+","")
    if '万' in x1:
        x1 = x1.replace('万','')
        x1 = float(x1) * 10000
        return int(x1)
    else:
        return int(x1)


df4 = pd.DataFrame()
df4['好评率'] = df3['好评率'].apply(process)
df4['好评数'] = df3['好评数'].apply(process)
df4['中评数'] = df3['中评数'].apply(process)
df4['差评数'] = df3['差评数'].apply(process)
df4['商品id'] = df3['商品id']

new_df = df4.groupby('商品id').agg('mean')
new_df['付款人数'] = new_df['好评数'] + new_df['中评数'] + new_df['差评数']

# 使用seaborn绘制多变量回归关系图
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(16,9),dpi=500)
sns.pairplot(new_df, kind='reg')
plt.savefig('多变量回归关系图.png')




