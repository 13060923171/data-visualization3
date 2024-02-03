import pandas as pd
import numpy as np
import datefinder
import re
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('movieinfo.xlsx')

def language_process(x):
    x1 = str(x)
    if "国语" in x1:
        return "国语"
    else:
        return "外语"

def time_process(x):
    x1 = str(x)
    if '-' in x1:
        matches = datefinder.find_dates(x1)
        for match in matches:
            date_str = match.strftime("%Y-%m-%d")
            return date_str
    else:
        return np.NAN

def categorize_date(date):
    # 把 12.15-2.15 归为一类
    if (date.month == 12 and date.day >= 15) or (date.month == 1) or (date.month == 2 and date.day <= 15):
        return '贺岁档'
    # 把 6.1-8.31 归为一类
    elif (date.month == 6 and date.day >= 1) or (date.month == 7) or (date.month == 8 and date.day <= 31):
        return '暑假档'
    # 其它日期归为其它类
    else:
        return '其他挡'

def movie_type(x):
    x1 = str(x).split(",")
    x1 = x1[0]
    if "剧情" in x1:
        return "剧情片"
    elif "喜剧" in x1:
        return "喜剧片"
    elif "动作" in x1:
        return "动作片"
    elif "爱情" in x1:
        return "爱情片"
    elif "冒险" in x1:
        return "冒险片"
    elif "惊悚" in x1:
        return "惊悚片"
    elif "奇幻" in x1:
        return "奇幻片"
    elif "犯罪" in x1:
        return "犯罪片"
    elif "科幻" in x1:
        return "科幻片"
    else:
        return "其他"

df['电影语言'] = df['电影语言'].apply(language_process)
df['电影上映时间'] = df['电影上映时间'].apply(time_process)
df['电影上映时间'] = pd.to_datetime(df['电影上映时间'])
df = df.dropna(subset=['电影上映时间','电影导演','主演'],axis=0)
df['time_category'] = df['电影上映时间'].apply(categorize_date)
df['电影类型'] = df['电影类型'].apply(movie_type)

d1 = {}
for d in df['电影导演']:
    d1[d] = d1.get(d,0) + 1

director = []
ls = list(d1.items())
ls.sort(key=lambda x:x[1],reverse=True)

for key,values in ls[:30]:
    director.append(key)

def director_process(x):
    x1 = str(x)
    if x1 in director:
        return 1
    else:
        return 0

df['电影导演'] = df['电影导演'].apply(director_process)


dic = {}
for d in df['主演']:
    d1 = str(d).split(";")
    for d2 in d1:
        d3 = str(d2).split("-")
        d4 = d3[0]
        dic[d4] = dic.get(d4,0) + 1


actor = []
ls1 = list(dic.items())
ls1.sort(key=lambda x:x[1],reverse=True)

for key,values in ls1[:30]:
    actor.append(key)


def actor_process1(x):
    list1 = []
    x1 = str(x).split(";")
    for d2 in x1:
        d3 = str(d2).split("-")
        d4 = d3[0]
        list1.append(d4)
    return " ".join(list1)

df['主演'] = df['主演'].apply(actor_process1)

# 构建正则表达式
regex = '|'.join(actor)
# 使用str.contains进行查找

df['热门主演'] = df['主演'].str.contains(regex)


def actor_process(x):
    if str(x) == 'True':
        return 1
    else:
        return 0

df['热门主演'] = df['热门主演'].apply(actor_process)

df = df.dropna(subset=['想看人数','总票房'],axis=0)

# 转换函数
def convert_to_num(s):
    if '亿' in s:
        return float(re.findall(r"\d+\.?\d*",s)[0]) * 1e8
    elif '万' in s:
        return float(re.findall(r"\d+\.?\d*",s)[0]) * 1e4
    else:
        return float(s)

df['总票房'] = df['总票房'].apply(convert_to_num)


# 创建 LabelEncoder 对象
label_encoder = LabelEncoder()
df['电影语言'] = label_encoder.fit_transform(list(df['电影语言']))
df['time_category'] = label_encoder.fit_transform(list(df['time_category']))
df['电影类型'] = label_encoder.fit_transform(list(df['电影类型']))
df['总票房（单位万）'] = df['总票房'] / 10000
df['总票房'] = df['总票房'].astype('float')

data = pd.DataFrame()
data['电影语言'] = df['电影语言']
data['上映时间'] = df['time_category']
data['电影类型'] = df['电影类型']
data['热门主演'] = df['热门主演']
data['电影导演'] = df['电影导演']
data['电影持续时间'] = df['电影持续时间']
data['想看人数'] = df['想看人数']
data['总票房（单位万）'] = df['总票房（单位万）']

# data.to_csv('特征数据.csv',encoding='utf-8-sig')

# 自变量
X = data.drop("总票房（单位万）", axis=1)
# 因变量
y = data["总票房（单位万）"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(np.array(y).reshape(-1, 1))

# 使用 30% 的数据进行测试，70% 的数据用于训练模型
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# 随机森林模型
model_rf = RandomForestRegressor(n_estimators=100)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# 将预测结果反归一化
y_pred_rf = y_scaler.inverse_transform(y_pred_rf.reshape(-1, 1))
y_test_rf = y_scaler.inverse_transform(y_test.reshape(-1, 1))

# 计算均方误差 (MSE)
rmse_rf = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
print("RMSE of RandomForestRegressor: ", rmse_rf)


# XGBRegressor模型
model_xgb = XGBRegressor(n_estimators=100)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

# 将预测结果反归一化
y_pred_xgb = y_scaler.inverse_transform(y_pred_xgb.reshape(-1, 1))
y_test_xgb = y_scaler.inverse_transform(y_test.reshape(-1, 1))

# 计算均方误差 (MSE)
rmse_xgb = np.sqrt(mean_squared_error(y_test_xgb, y_pred_xgb))
print("RMSE of XGBRegressor: ", rmse_xgb)

def plot_preds_vs_actuals(actuals, preds, model_name):
    fig, ax = plt.subplots()
    ax.scatter(actuals, preds)
    ax.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=4)
    ax.set_xlabel('Actuals')
    ax.set_ylabel('Predictions')
    ax.set_title(f'Actuals vs Predictions for {model_name}')
    plt.savefig(f'Actuals vs Predictions for {model_name}.png')
    plt.show()

#上述散点图中的黑色虚线代表理想的预测结果（预测值完全等于真实值），散点离这条黑线的距离代表预测误差的大小。如果预测准确的话，点应该大致分布在这条线的周围。
# 绘制对于随机森林的实际值vs预测值
plot_preds_vs_actuals(y_test_rf, y_pred_rf, 'RandomForestRegressor')

# 绘制对于XGBoost的实际值vs预测值
plot_preds_vs_actuals(y_test_xgb, y_pred_xgb, 'XGBRegressor')