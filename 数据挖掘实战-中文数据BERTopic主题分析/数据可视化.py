import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")

df = pd.read_excel('data_v5.xlsx')
df['数量'] = 1

def time_process1(x):
    try:
        x1 = str(x).split(" ")
        x1 = x1[0]
        if '2023年' not in x1:
            x1 = '2024年' + x1
        elif '2023年' in x1 and '07月' in x1:
            return np.NAN
        elif '2023年' in x1 and '08月' in x1:
            return np.NAN
        else:
            x1 = x1
        return x1
    except:
        return np.NAN


def time_process2(x):
    try:
        x1 = str(x).split(" ")
        x1 = x1[-1]
        x2 = str(x1).split(":")
        x2 = x2[0]
        return x2
    except:
        return np.NAN


def content_length(x):
    x1 = str(x).strip(" ")
    return len(x1)


df['日'] = df['博文发布时间'].apply(time_process1)
df['小时'] = df['博文发布时间'].apply(time_process2)
df['博文长度'] = df['清洗后内容'].apply(content_length)

df = df.dropna(subset=['日'],axis=0)
df['日'] = pd.to_datetime(df['日'], format='%Y年%m月%d日')
df = df.sort_values(by=['日'],ascending=True)

# 对数据按天进行重采样
daily_counts = df.groupby(by=['日']).agg({'数量':'count','博文长度':'sum'})

# df.index = df['日']
# weekly_counts = df.resample('W').count()


# def line1():
#     plt.figure()  # 创建一个新的图表窗口
#     new_df1 = weekly_counts['数量']
#     new_df1 = new_df1.sort_index()
#     x_data1 = [str(x).split(" ")[0] for x in new_df1.index]
#     y_data1 = [y for y in new_df1.values]
#     plt.figure(figsize=(20,9),dpi=500)
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.plot(x_data1,y_data1,color='#b82410',label='Week')
#     plt.legend()
#     plt.title('Posting Volume - Weekly Change Trend Chart')
#     plt.xlabel('Week')
#     plt.ylabel('Times')
#     plt.grid()
#     # 设置x轴标签倾斜45度
#     plt.xticks(rotation=45)
#     plt.savefig('./图片/每周变化趋势图.png')
#     plt.show()
#     new_df1.to_excel('./时间数据/每周变发帖数据.xlsx')
#
#
# def line2():
#     new_df1 = df['小时'].value_counts()
#     new_df1 = new_df1.sort_index()
#     x_data1 = [x for x in new_df1.index]
#     y_data1 = [y for y in new_df1.values]
#     plt.figure(figsize=(20,9),dpi=500)
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.plot(x_data1,y_data1,color='#b82410',label='hour')
#     plt.legend()
#     plt.title('The trend chart of 24-hour post volume changes')
#     plt.xlabel('hour')
#     plt.ylabel('postings/hour')
#     plt.grid()
#     plt.savefig('./图片/24小时发帖量变化趋势图.png')
#     plt.show()
#     new_df1.to_excel('./时间数据/小时发帖数据.xlsx')
#
#
def time_picture():
    # 对时间序列进行分解
    decomposition = seasonal_decompose(daily_counts['数量'], model='additive')
    # 进行ADF检验
    adf_result = adfuller(daily_counts['数量'])

    # 输出ADF检验结果
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print('\t%s: %.3f' % (key, value))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 绘制分解结果
    plt.figure(figsize=(14, 10),dpi=500)
    plt.subplot(411)
    plt.plot(decomposition.observed, label='Observed')
    plt.legend(loc='upper left')
    decomposition.observed.to_excel('./时间序列分解数据/Observed.xlsx')
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='upper left')
    decomposition.trend.to_excel('./时间序列分解数据/Trend.xlsx')
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    decomposition.seasonal.to_excel('./时间序列分解数据/Seasonal.xlsx')
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Random')
    plt.legend(loc='upper left')
    decomposition.resid.to_excel('./时间序列分解数据/Random.xlsx')
    plt.tight_layout()
    plt.savefig('./图片/时间序列分解图.png')
    plt.show()


# def time1():
#     # 对时间序列进行分解
#     decomposition = seasonal_decompose(daily_counts['数量'], model='additive')
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     matplotlib.rcParams['axes.unicode_minus'] = False
#     # 绘制分解结果
#     plt.figure(figsize=(14, 10), dpi=500)
#     plt.plot(decomposition.observed, label='Observed')
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#     plt.savefig('./图片/Observed.png')
#     plt.show()
#
# def time2():
#     # 对时间序列进行分解
#     decomposition = seasonal_decompose(daily_counts['数量'], model='additive')
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     matplotlib.rcParams['axes.unicode_minus'] = False
#     # 绘制分解结果
#     plt.figure(figsize=(14, 10), dpi=500)
#     plt.plot(decomposition.trend, label='Trend')
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#     plt.savefig('./图片/Trend.png')
#     plt.show()
#
# def time3():
#     # 对时间序列进行分解
#     decomposition = seasonal_decompose(daily_counts['数量'], model='additive')
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     matplotlib.rcParams['axes.unicode_minus'] = False
#     # 绘制分解结果
#     plt.figure(figsize=(14, 10), dpi=500)
#     plt.plot(decomposition.seasonal, label='Seasonal')
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#     plt.savefig('./图片/Seasonal.png')
#     plt.show()
#
# def time4():
#     # 对时间序列进行分解
#     decomposition = seasonal_decompose(daily_counts['数量'], model='additive')
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     matplotlib.rcParams['axes.unicode_minus'] = False
#     # 绘制分解结果
#     plt.figure(figsize=(14, 10), dpi=500)
#     plt.plot(decomposition.resid, label='Random')
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#     plt.savefig('./图片/Random.png')
#     plt.show()


# 线性回归分析
# def linear_regression_analysis(y_var, output_summary_path):
#     X = daily_counts.index.factorize()[0]  # 将日期因子化为数值
#     X = sm.add_constant(X)  # 添加常量项
#     y = daily_counts[y_var]
#
#     model = sm.OLS(y, X).fit()
#
#     summary = model.summary()
#
#     # 保存 summary 到文本文件
#     with open(output_summary_path, 'w') as f:
#         f.write(summary.as_text())
#
#     result = {
#         'Estimate': [model.params[1]],  # 线性模型的斜率
#         'SE': [model.bse[1]],  # 标准误差
#         't': [model.tvalues[1]],  # t 检验统计量
#         'R²': [model.rsquared],  # R² 值
#         'p-value': [model.pvalues[1]]  # p 值
#     }
#     print(result)
#     data1 = pd.DataFrame(result)
#     data1.to_excel(f'./线性相关数据/{y_var}-线性相关数值.xlsx')
#
#
# # 分别对 "数量" 和 "博文长度" 进行线性回归分析
# result_qty = linear_regression_analysis('数量', './线性相关数据/summary_发帖数量.txt')
# result_length = linear_regression_analysis('博文长度', './线性相关数据/summary_length.txt')




if __name__ == '__main__':
    # line1()
    # line2()
    time_picture()
    # time1()
    # time2()
    # time3()
    # time4()
