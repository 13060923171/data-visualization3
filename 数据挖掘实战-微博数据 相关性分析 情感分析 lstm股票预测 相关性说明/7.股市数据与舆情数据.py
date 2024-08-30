import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


df1 = pd.read_csv('./data/新_博文表.csv')
df2 = pd.read_excel('./data/股价表.xlsx')
df2['日期'] = pd.to_datetime(df2['日期'])
df2 = df2.sort_values('日期')
# 计算对数收益率
df2['对数收益率'] = np.log(df2['收盘价'] / df2['收盘价'].shift(1))
# 移除第一行，因为第一行的对数收益率是NaN
df2 = df2.dropna()
df2 = df2.set_index('日期')
# 将数据按月聚合，计算每月的标准差(波动率)
monthly_volatility = df2['对数收益率'].resample('M').std()



def time_process1(x):
    try:
        x1 = str(x).split(" ")
        x1 = x1[0]
        if '2023年' not in x1 and '2022年' not in x1:
            x1 = '2024年' + x1
        else:
            x1 = x1
        return x1
    except:
        return np.NAN


df1['博文发布时间'] = df1['博文发布时间'].apply(time_process1)
df1['博文发布时间'] = pd.to_datetime(df1['博文发布时间'],format='%Y年%m月%d日')
df1 = df1.sort_values(by=['博文发布时间'],ascending=True)
# 提取月份并创建一个新的列来存储月份信息
df1['博文发布日期'] = df1['博文发布时间'].dt.to_period('M')
df1['发帖数量'] = 1

# 按月份对互动量和情感热度进行聚合计算平均值
day_df1 = df1.groupby('博文发布日期').agg({'发帖数量': 'sum','情感得分': 'mean'})

data1 = pd.DataFrame()
data1['日期'] = day_df1['发帖数量'].index.astype(str)
data1['发帖数量'] = day_df1['发帖数量'].values
data1['情感得分'] = day_df1['情感得分'].values

data2 = pd.DataFrame()
data2['日期'] = monthly_volatility.index
data2['收益率'] = monthly_volatility.values


def time_date(x):
    x1 = str(x).split('-')
    x2 = x1[0] + "-" + x1[1]
    return x2


data2['日期'] = data2['日期'].apply(time_date)
data3 = pd.merge(data1,data2,on='日期')
data3.to_excel('舆情的数据与收盘价的关系原始数据.xlsx')


def picture1():
    # 创建一个图形和轴对象
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制发帖数量的柱状图
    ax1.plot(data3['日期'], data3['发帖数量'], color='skyblue', label='发帖数量', alpha=0.8)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('发帖数量')
    ax1.tick_params(axis='y')
    # 设置横坐标标签的显示间隔（例如每隔7个日期显示一个标签）
    # ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # 你可以调整nbins的值来控制显示的标签数量
    ax1.set_xticklabels(data3['日期'], rotation=45)

    # 创建第二个 y 轴，共享 x 轴
    ax2 = ax1.twinx()

    # 绘制收盘价的折线图
    ax2.plot(data3['日期'], data3['收益率'], color='orange', marker='o', label='收益率')
    ax2.set_ylabel('收益率')
    ax2.tick_params(axis='y')

    # 添加图例
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置图表标题
    plt.title('发帖数量与收益率的关系')

    # 自动调整布局
    plt.tight_layout()
    plt.savefig('发帖数量与收益率的关系.png')
    # 显示图表
    plt.show()


def picture2():
    # 创建一个图形和轴对象
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制发帖数量的柱状图
    ax1.plot(data3['日期'], data3['情感得分'], color='red', label='情感热度', alpha=0.8)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('情感热度')
    ax1.tick_params(axis='y')
    # 设置横坐标标签的显示间隔（例如每隔7个日期显示一个标签）
    # ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # 你可以调整nbins的值来控制显示的标签数量
    ax1.set_xticklabels(data3['日期'], rotation=45)

    # 创建第二个 y 轴，共享 x 轴
    ax2 = ax1.twinx()

    # 绘制收盘价的折线图
    ax2.plot(data3['日期'], data3['收益率'], color='orange', marker='o', label='收益率')
    ax2.set_ylabel('收益率')
    ax2.tick_params(axis='y')

    # 添加图例
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置图表标题
    plt.title('情感热度与收益率的关系')

    # 自动调整布局
    plt.tight_layout()
    plt.savefig('情感热度与收益率的关系.png')
    # 显示图表
    plt.show()


# def linear1():
#     # 计算发帖数量和收盘价、情感得分和收盘价之间的相关系数
#     corr_post_close = data3['发帖数量'].corr(data3['收盘价'])
#     corr_sent_close = data3['情感得分'].corr(data3['收盘价'])
#
#     print('发帖数量和收盘价之间的相关系数:', corr_post_close)
#     print('情感得分和收盘价之间的相关系数:', corr_sent_close)
#
#     # 准备数据
#     X = data3[['发帖数量', '情感得分']]
#     y = data3['收盘价']
#
#     # 训练线性回归模型
#     model = LinearRegression()
#     model.fit(X, y)
#
#     # 输出回归系数
#     print('回归系数:', model.coef_)
#     print('截距:', model.intercept_)
#
#     # 计算模型评分
#     score = model.score(X, y)
#     print('模型评分:', score)
#
#     # 查看每一项对收盘价的影响
#     coefficients = pd.DataFrame({
#         '特征': ['发帖数量', '情感得分'],
#         '回归系数': model.coef_
#     })
#
#     print(coefficients)
#
#
# def demo():
#     # 准备数据
#     X = data3[['发帖数量', '情感得分']]
#     y = data3['收盘价']
#     # 分割数据为训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # 初始化随机森林回归模型
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#
#     # 训练模型
#     model.fit(X_train, y_train)
#
#     # 预测
#     y_pred = model.predict(X_test)
#
#     # 计算模型评分和均方误差
#     r2 = r2_score(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#
#     print(f'R²: {r2}')
#     print(f'MSE: {mse}')
#
#     # 特征重要性
#     importance = model.feature_importances_
#     features = pd.DataFrame({
#         '特征': ['发帖数量', '情感得分'],
#         '重要性': importance
#     }).sort_values(by='重要性', ascending=False)
#
#     print(features)
#
if __name__ == '__main__':
    picture1()
    picture2()
    # linear1()
    # demo()