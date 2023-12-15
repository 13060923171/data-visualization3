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


if __name__ == '__main__':
    lda_strength()
    emotion_type()