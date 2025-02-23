import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')
sns.set_style(style="whitegrid")

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_excel('比特币数据.xlsx', parse_dates=['日期'], index_col='日期')
ts = data['收盘价'].resample('D').mean().ffill()
# 划分测试集（最后30天）
train = ts[:-30]
test = ts[-30:]

auto_model = auto_arima(
    train,
    start_p=0, max_p=10,
    start_q=0, max_q=10,
    d=1,  # 根据ADF检验结果设置
    seasonal=False,  # 不使用季节性
    trace=True,  # 打印搜索过程
    error_action='ignore',
    suppress_warnings=True
)

# ADF检验
result = adfuller(train)
print(f'ADF统计量: {result[0]:.3f}, p值: {result[1]:.3f}')

# 一阶差分（若p>0.05）
diff = train.diff().dropna()
adf_diff = adfuller(diff)
print(f'差分后ADF统计量: {adf_diff[0]:.3f}, p值: {adf_diff[1]:.3f}')

# 绘制差分序列的ACF和PACF图
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
plot_acf(diff, lags=20, ax=ax1)
plot_pacf(diff, lags=20, ax=ax2, method='ywm')
plt.savefig('acf_pacf.png')
plt.show()

# 假设通过分析确定参数为ARIMA(2,1,1)
model = SARIMAX(train, order=auto_model.order)
results = model.fit(disp=False)

# 预测未来30天
forecast = results.get_forecast(steps=len(test), dynamic=True)
pred_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(15,7))
plt.plot(train.index, train, label='历史数据', color='#1f77b4')
plt.plot(test.index, test, label='真实值', color='#2ca02c', linestyle='--')
plt.plot(pred_mean.index, pred_mean, label='预测值', color='#ff7f0e', marker='o')
plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1],
                 color='#ff7f0e', alpha=0.1)
plt.title('比特币收盘价ARIMA预测对比', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('收盘价（美元）', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('比特币收盘价ARIMA预测对比.png')
plt.show()

# 计算误差
mae = mean_absolute_error(test, pred_mean)
rmse = np.sqrt(mean_squared_error(test, pred_mean))
print(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}')

# 残差白噪声检验
residuals = results.resid
lb_test = acorr_ljungbox(residuals, lags=[10])
print(f'Ljung-Box检验p值: {lb_test.lb_pvalue.values[0]:.3f}')