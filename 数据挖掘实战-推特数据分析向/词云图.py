import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
import stylecloud

df1 = pd.read_csv('./data/0_20000_聚类结果.csv')
df2 = pd.read_csv('./data/20000_40000_聚类结果.csv')
df3 = pd.read_csv('./data/40000_60000_聚类结果.csv')
df4 = pd.read_csv('./data/60000_80000_聚类结果.csv')
df5 = pd.read_csv('./data/80000_100000_聚类结果.csv')
df6 = pd.read_csv('./data/100000_120000_聚类结果.csv')
df7 = pd.read_csv('./data/120000_140000_聚类结果.csv')
df8 = pd.read_csv('./data/140000_160000_聚类结果.csv')
df9 = pd.read_csv('./data/160000_180000_聚类结果.csv')
df10 = pd.read_csv('./data/180000_200000_聚类结果.csv')
data = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10],axis=0)
new_data = data.dropna(subset=["聚类结果"])


def main(number):
    df1 = new_data[new_data['聚类结果'] == number]
    stop_words = ['amp','bay','people','time','day','live','school','love','team','game','pm','st','la','week','tonight','green','fire','city','tampa']
    list_text = []
    for t in df1['new_推文']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            # 再过滤一遍无效词
            if i not in stop_words:
                # 添加到列表里面
                list_text.append(i)
    # 然后传入词云图中，筛选最多的100个词
    stylecloud.gen_stylecloud(text=' '.join(list_text), max_words=100,
                              # 不能有重复词
                              collocations=False,
                              max_font_size=400,
                              # 字体样式
                              font_path='simhei.ttf',
                              # 图片形状
                              icon_name='fas fa-crown',
                              # 图片大小
                              size=1200,
                              # palette='matplotlib.Inferno_9',
                              # 输出图片的名称和位置
                              output_name='./data/聚类{}-词云图.png'.format(number))
    # 开始生成图片
    Image(filename='./data/聚类{}-词云图.png'.format(number))


if __name__ == '__main__':
    for i in range(0,7):
        main(i)