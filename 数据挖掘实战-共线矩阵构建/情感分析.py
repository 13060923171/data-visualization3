import pandas as pd
import nltk
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# df1 = pd.read_csv('sum_comment.csv',encoding="utf-8-sig")
#
#
# sid = SentimentIntensityAnalyzer()
#
# f = open('C-class-fenci.txt', 'w', encoding='utf-8')
# for line in df1['comment']:
#     tokens = nltk.word_tokenize(line)
#     # 计算关键词
#     all_words = tokens
#     c = Counter()
#     for x in all_words:
#         if len(x) > 1 and x != '\r\n':
#             c[x] += 1
#     # Top50
#     output = ""
#     # print('\n词频统计结果：')
#     for (k, v) in c.most_common(30):
#         # print("%s:%d"%(k,v))
#         output += k + " "
#
#     f.write(output + "\n")
#
# else:
#     f.close()
#
#
#
# sum_counts = 0
# text_list = []
#
#
# def emotional_judgment(x):
#     neg = x['neg']
#     neu = x['neu']
#     pos = x['pos']
#     compound = x['compound']
#     if compound == 0 and neg == 0 and pos == 0 and neu == 1:
#         return 'neu'
#     if compound > 0:
#         if pos > neg:
#             return 'pos'
#         else:
#             return 'neg'
#     elif compound < 0:
#         if pos < neg:
#             return 'neg'
#         else:
#             return 'pos'
#
#
# df1['scores'] = df1['comment'].apply(lambda commentText: sid.polarity_scores(commentText))
# df1['compound'] = df1['scores'].apply(lambda score_dict: score_dict['compound'])
# df1['Negtive'] = df1['scores'].apply(lambda score_dict: score_dict['neg'])
# df1['Postive'] = df1['scores'].apply(lambda score_dict: score_dict['pos'])
# df1['Neutral'] = df1['scores'].apply(lambda score_dict: score_dict['neu'])
# df1['comp_score'] = df1['scores'].apply(emotional_judgment)
# df1.to_csv('emotional_comment.csv',encoding="utf-8-sig")


df = pd.read_csv('emotional_comment.csv')
new_df = df['comp_score'].value_counts()
x_data1 = new_df.index
y_data1 = new_df.values
print(x_data1)
print(y_data1)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(9, 6))  # 调节图形大小
labels = x_data1  # 定义标签
sizes = y_data1 # 每块值
colors = ['#2E86C1', '#5DADE2','#EC7063']  # 每块颜色定义
explode = (0, 0.05,0)  # 将某一块分割出来，值越大分割出的间隙越大
patches, text1, text2 = plt.pie(sizes,
                                explode=explode,
                                labels=labels,
                                colors=colors,
                                labeldistance=1.1,  # 图例距圆心半径倍距离
                                autopct='%3.2f%%',  # 数值保留固定小数位
                                shadow=False,  # 无阴影设置
                                startangle=90,  # 逆时针起始角度设置
                                pctdistance=0.6)  # 数值距圆心半径倍数距离
# patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
# x，y轴刻度设置一致，保证饼图为圆形
plt.axis('equal')
plt.legend()
plt.title('情感占比分析')
plt.savefig('情感占比分析.png')
plt.show()