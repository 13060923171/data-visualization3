import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from pmdarima import auto_arima
from statsmodels.graphics.api import qqplot
import os
matplotlib.use('Agg')  # 使用Agg后端


def demo():
    df = pd.read_csv('data.csv')
    def datetime1(x):
        x = str(x).replace('-', '/').split(" ")
        return x[0]

    df['日期'] = df['日期'].apply(datetime1)
    # 如果日期列是字符串，使用pd.to_datetime转换，并指定格式
    date_format = "%Y/%m/%d"
    df.index = pd.to_datetime(df['日期'], format=date_format)
    # 按日期排序
    data = df.sort_index()
    #对时间列进行归类，按周来分布，找出每周发帖的总数
    data_month = data['发帖数量'].resample("W").sum()
    df1 = pd.DataFrame()
    df1['time'] = data_month.index
    df1['value'] = data_month.values

    #对数据进行一阶差分处理
    data1 = df1['value'].diff().dropna()

    #自动去找出最优模型
    best_fit = auto_arima(data1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    #根据前面找到的最优数值来输出最优模型
    model = SARIMAX(data1, order=best_fit.order, seasonal_order=best_fit.seasonal_order)
    #对模型进行拟合
    model_fit = model.fit(disp=False)
    #画图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 在选出模型检验之后，对arma模型所产生的残差做相关图
    resid = model_fit.resid
    #画出正态分布趋势
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)
    plt.title('Normal distribution plot')
    plt.savefig('./img/Normal distribution plot.png')

    # 查看拟合效果
    delta = model_fit.fittedvalues - data1
    # 它的值在0-1之间，越接近1，拟合效果越好
    score = 1 - delta.var() / data1.var()
    with open('./data/拟合得分.txt', 'w', encoding='utf-8-sig') as f:
        f.write("拟合得分为：{}".format(score))

    # 预测后面时间发布频率的趋势，预测接下来五周后的发帖趋势
    prediction = model_fit.get_prediction(start=list(data1.index)[-1], end=len(data1) + 5, dynamic=False,
                                          full_results=True)
    predicted_mean = prediction.predicted_mean
    predicted_confidence_intervals = prediction.conf_int()
    # 画图
    plt.figure(figsize=(15, 9))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 原先的数值图
    plt.plot(data1, label='原始数据')
    # 预测的数值图
    plt.plot(predicted_mean.index, predicted_mean, color='r', label='预测数据')
    # 把两者结合在一起，生成预测效果图
    plt.fill_between(predicted_confidence_intervals.index, predicted_confidence_intervals.iloc[:, 0],
                     predicted_confidence_intervals.iloc[:, 1], color='pink')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.title('time prediction')
    plt.savefig('./img/time prediction.png')
    plt.show()


if __name__ == '__main__':
    demo()

