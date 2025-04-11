import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')  # 使用Agg后端
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_excel('new_data.xlsx')
df['发布时间'] = pd.to_datetime(df['发布时间'])
df['month'] = df['发布时间'].dt.to_period('M')
df = df.sort_values(by=['month'],ascending=True)
df = df.dropna(subset=['month'],axis=0)


def data_process(x):
    if '万' in str(x):
        x1 = str(x).replace("万","")
        x2 = int(float(x1) * 10000)
        return x2
    elif 'nan' in str(x):
        return 0
    else:
        return int(x)

df['点赞数'] = df['点赞数'].apply(data_process)
df['评论数'] = df['评论数'].apply(data_process)
df['分享数'] = df['分享数'].apply(data_process)
df['笔记热度'] = df['点赞数'] + df['评论数'] + df['分享数']

new_df = df.groupby(['month'])['笔记热度'].mean().reset_index()
new_df = new_df.iloc[7:]


# 生成完整的月份范围
start_month = new_df['month'].min()
end_month = new_df['month'].max()
all_months = pd.period_range(start=start_month, end=end_month, freq='M')

# 创建完整月份的DataFrame并合并
full_months_df = pd.DataFrame({'month': all_months})
merged_df = full_months_df.merge(new_df, on='month', how='left').fillna({'笔记热度': 0})

# 转换month列为字符串格式（可选）
merged_df['month'] = merged_df['month'].astype(str)

# 按月份排序
merged_df = merged_df.sort_values('month').reset_index(drop=True)

merged_df['笔记热度'] = merged_df['笔记热度'].astype(int)

if not os.path.exists("./ARIMA_DATA"):
    os.mkdir("./ARIMA_DATA")

merged_df.to_excel('./ARIMA_DATA/月份热度原始数据.xlsx',index=False)

def model(merged_df):
    # 数据预处理（使用你处理好的merged_df）
    # 转换为时间序列索引
    ts = merged_df.copy()
    ts['month'] = ts['month'].apply(lambda x: pd.Period(x))  # 确保转换为Period类型
    ts = ts.set_index('month').to_timestamp(freq='M')['笔记热度']

    # 1. 一阶差分处理
    diff = ts.diff().dropna()

    # 平稳性检验
    def adf_test(timeseries):
        print('ADF检验结果:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        dfoutput.to_excel('./ARIMA_DATA/ADF检验结果.xlsx')
        print(dfoutput)

    adf_test(diff)

    plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示
    # 可视化ACF/PACF
    plt.figure(figsize=(12, 6))
    plot_acf(diff, lags=12, ax=plt.subplot(121))
    plot_pacf(diff, lags=12, ax=plt.subplot(122))
    plt.tight_layout()
    plt.savefig('./ARIMA_DATA/ACF PACF.png',dpi=300)
    plt.show()

    # 网格搜索寻找最优参数
    best_aic = np.inf
    best_order = None

    # 参数搜索范围
    for p in range(3):
        for q in range(3):
            try:
                model = ARIMA(ts, order=(p, 1, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, 1, q)
            except:
                continue

    print(f'最优参数组合：ARIMA{best_order}，AIC值：{best_aic:.2f}')

    # 使用最优模型拟合
    final_model = ARIMA(ts, order=best_order)
    final_results = final_model.fit()

    # 模型诊断
    print(final_results.summary())
    final_results.plot_diagnostics(figsize=(20, 12))
    plt.savefig('./ARIMA_DATA/模型诊断.png',dpi=300)
    plt.show()

    # 进行预测
    forecast_steps = 1  # 预测25年5月
    forecast = final_results.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps + 1, freq='M')[1:]  # 生成预测日期

    # 可视化结果
    plt.figure(figsize=(14, 7))
    plt.plot(ts.index, ts, label='实际值')
    plt.plot(forecast_index, forecast.predicted_mean,
             'ro', markersize=8, label='预测值')

    # 添加置信区间
    plt.fill_between(forecast_index,
                     forecast.conf_int().iloc[:, 0],
                     forecast.conf_int().iloc[:, 1],
                     color='r', alpha=0.1)

    plt.title('笔记热度时间序列预测(ARIMA模型)')
    plt.xlabel('日期')
    plt.ylabel('笔记热度')
    plt.legend()
    plt.savefig('./ARIMA_DATA/笔记热度时间序列预测.png',dpi=300)
    plt.grid(True)

    # 标注预测数值
    pred_value = int(forecast.predicted_mean[0])
    # plt.annotate(f'2025-05预测值：{pred_value}',
    #              xy=(forecast_index[0], pred_value),
    #              xytext=(forecast_index[0] - pd.DateOffset(months=6), pred_value * 1.1),
    #              arrowprops=dict(facecolor='black', shrink=0.05))
    # plt.show()

    # 输出预测结果
    print(f'预测2025年5月的平均笔记热度为：{pred_value}')


if __name__ == '__main__':
    model(merged_df)