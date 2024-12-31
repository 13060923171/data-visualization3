import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")
matplotlib.rcParams['axes.unicode_minus']=False

data = {
    '市场利率R': [0.0345, 0.0345, 0.0345, 0.0345, 0.0345, 0.0345, 0.0335, 0.0335, 0.0335, 0.031, 0.031, 0.031],
    '余额宝收益率': [0.01952, 0.0196, 0.0192, 0.0184, 0.0172, 0.0184, 0.0151, 0.0143, 0.0145, 0.0142, 0.0144, 0.0136],
    '资金规模S': [714.84, 720.22, 727.6, 798.64, 862.87, 913.07, 886.78, 858.46, 839.35, 825.78, 800.76, 789.98],
    '存款准备金率': [0.071, 0.071, 0.071, 0.071, 0.071, 0.071, 0.071, 0.071, 0.066, 0.066, 0.066, 0.066],
    '上证指数': [2788, 2921, 3022, 3086, 3126, 2984, 2928, 2919, 3287, 3386, 3392, 3401]
}

df = pd.DataFrame(data, index=pd.date_range(start='2024-01-01', periods=12, freq='M'))

# 选择适当的滞后阶数
model = VAR(df)
lags = model.select_order(maxlags=1)
print(lags.summary())

# 拟合VAR模型
var_model = model.fit(maxlags=lags.selected_orders['aic'], ic='aic')
print(var_model.summary())

# 计算脉冲响应函数
irf = var_model.irf(12)  # 选择12期的脉冲响应


# 绘制脉冲响应函数
plt.figure(figsize=(16, 9))
plt.rcParams['font.sans-serif'] = ['SimHei']

irf.plot(orth=False)
plt.savefig('脉冲响应.png')
plt.show()

# 绘制市场利率对余额宝收益率的脉冲响应
plt.figure(figsize=(14, 8))
irf.plot(orth=False, impulse='市场利率R', response='余额宝收益率')
plt.title('市场利率对余额宝收益率的脉冲响应')
plt.savefig('市场利率对余额宝收益率的脉冲响应.png')
plt.show()

# 绘制资金规模对余额宝收益率的脉冲响应
plt.figure(figsize=(14, 8))
irf.plot(orth=False, impulse='资金规模S', response='余额宝收益率')
plt.title('资金规模对余额宝收益率的脉冲响应')
plt.savefig('资金规模对余额宝收益率的脉冲响应.png')
plt.show()

# 绘制存款准备金率对余额宝收益率的脉冲响应
plt.figure(figsize=(14, 8))
irf.plot(orth=False, impulse='存款准备金率', response='余额宝收益率')
plt.title('存款准备金率对余额宝收益率的脉冲响应')
plt.savefig('存款准备金率对余额宝收益率的脉冲响应.png')
plt.show()

# 绘制上证指数对余额宝收益率的脉冲响应
plt.figure(figsize=(14, 8))
irf.plot(orth=False, impulse='上证指数', response='余额宝收益率')
plt.title('上证指数对余额宝收益率的脉冲响应')
plt.savefig('上证指数对余额宝收益率的脉冲响应.png')
plt.show()

# Granger因果检验
granger_test = var_model.test_causality('余额宝收益率', ['市场利率R', '资金规模S', '存款准备金率', '上证指数'], kind='f')
print(granger_test)

# 方差分解
fevd = var_model.fevd(12)
plt.figure(figsize=(20, 14))
# 获取方差分解的结果
fevd_results = fevd.plot()

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.4, wspace=0.4)

plt.suptitle('方差分解', fontsize=16)  # 设置总标题
plt.savefig('方差分解.png')
plt.show()