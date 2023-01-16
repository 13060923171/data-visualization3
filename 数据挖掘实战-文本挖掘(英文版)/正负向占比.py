import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('new_data.csv')
comp_score = df['comp_score'].value_counts()
x_data = list(comp_score.index)
y_data = list(comp_score.values)


plt.figure(figsize=(9,6),dpi=300)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(12, 9))  # 调节图形大小
labels = x_data  # 定义标签
sizes = y_data  # 每块值
colors = ['#58D68D','#45B39D','#1ABC9C']  # 每块颜色定义
patches, text1, text2 = plt.pie(sizes,
                                labels=labels,
                                colors=colors,
                                labeldistance=1.1,  # 图例距圆心半径倍距离
                                autopct='%3.1f%%',  # 数值保留固定小数位
                                shadow=False,  # 无阴影设置
                                startangle=90,  # 逆时针起始角度设置
                                pctdistance=0.6)  # 数值距圆心半径倍数距离
# patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
# x，y轴刻度设置一致，保证饼图为圆形
plt.axis('equal')
plt.legend()
plt.title('情感占比')
plt.savefig('情感占比.jpg')
plt.show()



