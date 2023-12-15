import pandas as pd
# 数据处理库
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style(style="whitegrid")

def lda_strength():
    df = pd.read_csv('特征词.csv')

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)

    x_data = [str(x).replace('Topic', '') for x in df['所属主题']]
    y_data = list(df['文章数量'])
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('theme strength')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('theme strength.png')


def emotion_type():
    df = pd.read_csv('new_data.csv')
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(dpi=500)
    new_df = df['情感分类'].value_counts()
    x_data = [str(x) for x in new_df.index]
    y_data = [int(x) for x in new_df.values]

    new_df.to_csv('情感分类数据.csv',encoding='utf-8-sig')
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('Sentiment classification')
    # 添加图例
    plt.legend(x_data, loc='lower right')
    plt.tight_layout()
    plt.savefig('Sentiment classification.png')


def lda_emotion_type():
    df1 = pd.read_csv('new_data.csv')
    df2 = pd.read_csv('lda_data.csv')
    df = pd.concat([df1,df2],axis=1)
    df3 = df.drop(['评论','分词'],axis=1)

    def demo(x):
        df4 = x
        new_df = df4['情感分类'].value_counts()
        x_data = [str(x) for x in new_df.index]
        y_data = [int(x) for x in new_df.values]
        d = {}
        for x, y in zip(x_data,y_data):
            d[x] = y
        return d

    new_df1 = df3.groupby('主题类型').apply(demo)
    new_df1.to_csv('主题特征情感分类情况.csv',encoding='utf-8-sig')


if __name__ == '__main__':
    # lda_strength()
    # emotion_type()
    lda_emotion_type()