import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stylecloud
from IPython.display import Image
from tqdm import tqdm

def main():
    df = pd.read_csv('./B站数据/B站数据.csv')
    d = {}
    list_text = []
    for t in df['分词']:
        # 把数据分开
        t = str(t).split(" ")
        for i in t:
            # 添加到列表里面
            list_text.append(i)
            d[i] = d.get(i,0)+1

    ls = list(d.items())
    ls.sort(key=lambda x:x[1],reverse=True)
    x_data = []
    y_data = []
    for key,values in ls[:200]:
        x_data.append(key)
        y_data.append(values)

    data = pd.DataFrame()
    data['word'] = x_data
    data['counts'] = y_data

    data.to_csv('./B站数据/高频词Top200.csv',encoding='utf-8-sig',index=False)
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
                              output_name='./B站数据/词云图.png')
    # 开始生成图片
    Image(filename='./B站数据/词云图.png')


    new_data = df['情感类型'].value_counts()
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    x_data = list(new_data.index)
    y_data = list(new_data.values)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('情感分布情况')
    plt.tight_layout()
    plt.savefig('./B站数据/情感分布情况.png')


if __name__ == '__main__':
    main()