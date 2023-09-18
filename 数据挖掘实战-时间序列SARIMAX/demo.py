import numpy as np
import pandas as pd
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


def demo(name):
    #创建文件夹
    if not os.path.exists("./{}".format(name)):
        os.mkdir("./{}".format(name))
    #找出关键词行
    data = df[df['关键词en'] == name]
    #删除重复项
    data = data.drop_duplicates(subset='博文id',keep='first')
    data['发帖数量'] = 1
    #把日期转化为时间序列
    data['发表日期'] = pd.to_datetime(data['发表日期'])
    data.index = data['发表日期']
    #对时间列进行归类，按月来分布，找出每个月发帖的总数
    data_month = data.resample("M").sum()
    print(data_month)

    #对数据进行一阶差分处理
    data1 = data_month['发帖数量'].diff().dropna()
    data1 = data1.diff().dropna()

    #判断data1这个长度是否符合，不符合就排除
    if len(data1) >= 2:
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
        plt.savefig('./{}/Normal distribution plot.png'.format(name))

        # 查看拟合效果
        delta = model_fit.fittedvalues - data1
        # 它的值在0-1之间，越接近1，拟合效果越好
        score = 1 - delta.var() / data1.var()
        with open('./{}/拟合得分.txt'.format(name),'w',encoding='utf-8-sig')as f:
            f.write("拟合得分为：{}".format(score))

        # 预测后面时间发布频率的趋势，预测接下来五个月的发帖趋势
        prediction = model_fit.get_prediction(start=list(data1.index)[-1], end=len(data1)+5, dynamic=False, full_results=True)
        predicted_mean = prediction.predicted_mean
        predicted_confidence_intervals = prediction.conf_int()
        #画图
        plt.figure(figsize=(15,9))
        #原先的数值图
        plt.plot(data1, label='Observed')
        #预测的数值图
        plt.plot(predicted_mean.index, predicted_mean, color='r', label='Forecast')
        #把两者结合在一起，生成预测效果图
        plt.fill_between(predicted_confidence_intervals.index, predicted_confidence_intervals.iloc[:, 0], predicted_confidence_intervals.iloc[:, 1], color='pink')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.legend()
        plt.title('time prediction')
        plt.savefig('./{}/time prediction.png'.format(name))
        plt.show()
    else:
        print('空值')


if __name__ == '__main__':
    #读取文件
    df = pd.read_excel('处理结果.xlsx')
    #判断关键词的数量
    new_df = df['关键词en'].value_counts()
    #生成关键词种类
    x_data = list(new_df.index)
    for x in tqdm(x_data):
        demo(x)

