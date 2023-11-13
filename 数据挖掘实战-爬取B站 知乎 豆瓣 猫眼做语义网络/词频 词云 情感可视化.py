import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stylecloud
from IPython.display import Image
from tqdm import tqdm

def main(name):
    df = pd.read_csv('./数据集/{}.csv'.format(name))
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

    data.to_csv('./词频/{}_高频词Top200.csv'.format(name),encoding='utf-8-sig',index=False)
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
                              output_name='./词云图/{}-词云图.png'.format(name))
    # 开始生成图片
    Image(filename='./词云图/{}-词云图.png'.format(name))


    new_data = df['情感类型'].value_counts()
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    x_data = list(new_data.index)
    y_data = list(new_data.values)
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('{}_情感分布情况'.format(name))
    plt.tight_layout()
    plt.savefig('./情感_可视化/{}_情感分布情况.png'.format(name))


if __name__ == '__main__':
    list_name = ['阿凡达2：水之道', '速度与激情10', '银河护卫队3', '奥本海默', '变形金刚7：超能勇士崛起', '蜘蛛侠：纵横宇宙', '蚁人3', '芭比', '超级马里奥', '黑豹2',
                 '闪电侠', '龙与地下城', '雷霆沙赞', '疯狂元素城', '小美人鱼',
                 '阿凡达1', '速度与激情9', '银河护卫队2', '信条', '大黄蜂', '蜘蛛侠：平行宇宙', '蚁人2', '黑豹', '神奇女侠1984', '疯狂动物城', '美女与野兽',
                 '长津湖', '战狼2', '你好李焕英', '哪吒之魔童降世', '流浪地球', '满江红',
                 '唐人街探案3', '复仇者联盟4终局之战', '长津湖之水门桥', '流浪地球2', '红海行动']

    for l in tqdm(list_name):
        main(l)