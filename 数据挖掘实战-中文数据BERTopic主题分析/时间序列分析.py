import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


# df1 = pd.read_csv('./bertopic_data/聚类结果.csv')
# df2 = pd.read_excel('data_v6.xlsx')
#
#
# df3 = pd.merge(df1,df2,left_on='Document',right_on='分词')
# df3.to_excel('data_v7.xlsx',index=False)

df = pd.read_excel('data_v7.xlsx')

# 将 'Date' 列设置为索引
df['日'] = pd.to_datetime(df['日'])
df['周'] = df['日'].dt.to_period('W')
df.set_index('周', inplace=True)

# 如果数据不平稳，先进行差分
df['Topic_diff'] = df['Topic'].diff().dropna()
# 可视化差分后的数据
df['Topic_diff'].plot(figsize=(10, 6))
plt.title('Differenced Topic Data')
plt.savefig('Differenced Topic Data.png')
plt.show()


# # 拟合ARIMA模型
# model = ARIMA(df['Topic_diff'].dropna(), order=(1, 1, 1))  # p=1, d=1, q=1
# model_fit = model.fit()
#
# # 打印模型总结信息
# print(model_fit.summary())
#
#
# # 预测未来10周的值
# forecast_steps = 10
# forecast = model_fit.forecast(steps=forecast_steps)
#
# # 创建未来的时间索引
# future_index = pd.period_range(start=df.index[-1] + 1, periods=forecast_steps, freq='W')
#
# # 将预测结果转换为Series
# forecast_series = pd.Series(forecast, index=future_index)
#
# # 可视化实际数据和预测数据
# plt.figure(figsize=(10, 6))
# plt.plot(df.index, df['Topic_diff'], label='Actual Data')
# plt.plot(future_index, forecast_series, label='Forecast', color='red')
# plt.title('ARIMA Model Forecast')
# plt.legend()
# plt.show()