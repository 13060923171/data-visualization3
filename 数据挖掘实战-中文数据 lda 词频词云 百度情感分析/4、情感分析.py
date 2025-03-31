import pandas as pd
import numpy as np
from paddlenlp import Taskflow
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端


# def emotion_analysis(text):
#     # 进行情感分析
#     results = sentiment_analysis(text)
#     for result in results:
#         label = result['label']
#         score = result['score']
#         if label == 'negative' and score <= 0.6:
#             label_class = '中性反馈'
#         elif label == 'negative' and 0.6 < score <= 0.7:
#             label_class = '轻度不满'
#         elif label == 'negative' and 0.7 < score <= 0.8:
#             label_class = '中度不满'
#         elif label == 'negative' and 0.8 < score <= 0.9:
#             label_class = '重度不满'
#         elif label == 'negative' and 0.9 < score <= 10:
#             label_class = '极度不满'
#         if label == 'positive' and score <= 0.6:
#             label_class = '中性反馈'
#         elif label == 'positive' and 0.6 < score <= 0.7:
#             label_class = '轻度喜欢'
#         elif label == 'positive' and 0.7 < score <= 0.8:
#             label_class = '中度喜欢'
#         elif label == 'positive' and 0.8 < score <= 0.9:
#             label_class = '重度喜欢'
#         elif label == 'positive' and 0.9 < score <= 10:
#             label_class = '极度喜欢'
#         return label,score,label_class
#
#
# df = pd.read_csv('./lda/lda_data.csv')
# # 初始化情感分析任务
# sentiment_analysis = Taskflow("sentiment_analysis")
# list_label = []
# list_score = []
# list_class = []
# for d in df['fenci']:
#     label,score,label_class = emotion_analysis(d)
#     list_label.append(label)
#     list_score.append(score)
#     list_class.append(label_class)
# df['label'] = list_label
# df['score'] = list_score
# df['class'] = list_class
# df.to_excel('情感分析.xlsx')

def emotion_pie(df1,name):
    new_df = df1['class'].value_counts()
    x_data = [x for x in new_df.index]
    y_data = [y for y in new_df.values]
    plt.figure(figsize=(16, 9), dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title(f'{name}-情感占比分布')
    plt.tight_layout()
    # 添加图例
    plt.legend(x_data, loc='upper left')
    plt.savefig(f'{name}-情感占比分布.png')


df = pd.read_excel('情感分析.xlsx')

list_number = [[0,2,9,10,13],[1,4,6,12],[3,7,11],[5,8,11]]
list_name =['价格诚信','物流售后','平台责任','商品质量']
for n1,n2 in zip(list_number,list_name):
    df1 = df[df['主题类型'].isin(n1)]
    emotion_pie(df1,n2)

