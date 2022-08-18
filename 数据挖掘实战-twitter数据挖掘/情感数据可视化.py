import pandas as pd
import matplotlib.pyplot as plt
#读取数据
df = pd.read_csv('新_第二组.csv')
#统计正面负面中立的数量
comp_score = df['comp_score'].value_counts()
x_data = list(comp_score.index)
y_data = list(comp_score.values)

#设定图片的风格
plt.style.use('fast')
#绘制柱状图
plt.figure(figsize=(9,6),dpi=300)
#采用柱状图的格式
plt.bar(x_data,y_data,color='#3498DB')
#可以正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
#图片标题
plt.title("第二组_情感占比")
#x轴标签
plt.xlabel("类别")
#y轴标签
plt.ylabel("数量")
#保存图片为png格式
plt.savefig('第二组_情感占比.png')
plt.show()



