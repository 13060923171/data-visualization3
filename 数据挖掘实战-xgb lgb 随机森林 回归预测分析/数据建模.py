import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score
#管道
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('car_sum.csv')
print(df.info())

def process(x):
    x1 = str(x)
    if x1 == '生产方式':
        return np.NAN
    else:
        return x1
df['生产方式'] = df['生产方式'].apply(process)
df = df.drop(['变速箱'],axis=1)
df = df.dropna(how='any',axis=0)

def pie1():
    new_df = df['品牌'].value_counts()
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16, 9), dpi=500)

    x_data = [x for x in new_df.index][:5]
    y_data = [x for x in new_df.values][:5]
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('最热门的Top5 品牌')
    plt.legend(x_data, loc='upper left')
    plt.tight_layout()
    plt.savefig('最热门的Top5 品牌.png')
    plt.show()

def bar1():
    new_df = df['生产方式'].value_counts()
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16, 9), dpi=500)
    x_data = [x for x in new_df.index]
    y_data = [x for x in new_df.values]
    plt.bar(x_data,y_data)
    plt.title("合资 VS 自主")
    plt.xlabel("数量")
    plt.ylabel("生成方式")
    plt.savefig('合资 VS 自主.png')
    plt.show()

def bar2():
    new_df = df['过户次数'].value_counts()
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16, 9), dpi=500)

    x_data = [x for x in new_df.index]
    y_data = [x for x in new_df.values]
    plt.bar(x_data,y_data)
    plt.title("过户次数分布趋势")
    plt.xlabel("数量")
    plt.ylabel("次数")
    plt.savefig('过户次数分布趋势.png')
    plt.show()

def process2(x):
    x1 = str(x).replace('次','')
    x2 = int(x1)
    if x2 >=3:
        return '3次及以上'
    else:
        return str(x2) + "次"

df['过户次数'] = df['过户次数'].apply(process2)

def process3(x):
    x1 = str(x).split("+")
    x2 = x1[0]
    if x2 != '汽油' and x2 != '纯电动':
        return '混合动力'
    else:
        return x2

df['能源形式'] = df['能源形式'].apply(process3)

def pie2():
    new_df = df['能源形式'].value_counts()
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16, 9), dpi=500)

    x_data = [x for x in new_df.index][:5]
    y_data = [x for x in new_df.values][:5]
    plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
    plt.title('能源形式 占比')
    plt.legend(x_data, loc='upper left')
    plt.tight_layout()
    plt.savefig('能源形式 占比.png')
    plt.show()

def line1():
    new_df = df['上牌时间'].value_counts()
    new_df = new_df.sort_index()
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16, 9), dpi=500)

    x_data = [x for x in new_df.index]
    y_data = [x for x in new_df.values]
    plt.plot(x_data,y_data,color='#b82410',label='上牌时间')
    plt.title('上牌时间分布趋势')
    plt.legend(x_data, loc='upper left')
    plt.tight_layout()
    plt.savefig('上牌时间分布趋势.png')
    plt.show()

def process4(x):
    try:
        x1 = str(x).replace('万公里','').replace('O','0').strip(" ")
        x1 = x1.strip("\n")
        x1 = x1.strip("\t")
        x1 = x1.strip("\r")
        x1 = float(x1)
        return x1
    except:
        return np.NAN

def process5(x):
    try:
        x1 = str(x).replace('万','').replace('O','0').strip(" ")
        x1 = x1.strip("\n")
        x1 = x1.strip("\t")
        x1 = x1.strip("\r")
        x1 = float(x1)
        return x1
    except:
        return np.NAN

def process6(x):
    x1 = str(x).replace('年','')
    return int(x1)

df['表显里程'] = df['表显里程'].apply(process4)
df['价格'] = df['价格'].apply(process5)
df['厂商指导价(万元)'] = df['厂商指导价(万元)'].apply(process5)
df['上牌时间'] = df['上牌时间'].apply(process6)
df = df.dropna(how='any',axis=0)

le = LabelEncoder()
df['生产方式'] = le.fit_transform(df['生产方式'])
df['过户次数'] = le.fit_transform(df['过户次数'])
df['能源形式'] = le.fit_transform(df['能源形式'])
df['整车质保'] = le.fit_transform(df['整车质保'])
df['车身形式'] = le.fit_transform(df['车身形式'])
df['厂商'] = le.fit_transform(df['厂商'])


def box_plot():
    data1 = list(df['价格'])
    data2 = list(df['表显里程'])
    data3 = list(df['厂商指导价(万元)'])
    data = [data2,data1,data3]
    y = ['表显里程','价格','厂商指导价(万元)']

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置箱线图的参数
    fig, ax = plt.subplots()
    ax.boxplot(data)

    # 设置x轴标签
    plt.xticks([1, 2, 3], y)

    # 设置y轴标签
    plt.ylabel('单位:/万')

    # 设置标题
    plt.title('箱线图数据展示')

    plt.savefig('箱线图数据展示.png')


def pair_plot():
    data = pd.DataFrame()
    data['价格'] = df['价格']
    data['厂商指导价(万元)'] = df['厂商指导价(万元)']
    data['表显里程'] = df['表显里程']
    # 使用seaborn绘制多变量回归关系图
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16,9),dpi=500)
    sns.pairplot(data, kind='reg')
    plt.savefig('多变量回归关系图.png')


df = df.drop(['车辆名称','品牌','上市时间','电动机总功率(kW)'],axis=1)


def hea_tmap():
    data = df
    # 使用corr函数计算相关性矩阵
    correlation_matrix = data.corr()
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16, 9), dpi=500)
    # 使用seaborn库绘制相关性矩阵的热力图
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")

    # 显示图形
    plt.savefig('相关性矩阵.png')


def Modeling_():
    data = df[['上牌时间','表显里程','厂商','生产方式','厂商指导价(万元)','能源形式','车身形式','整车质保','过户次数']]
    targe = df['价格']
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data, targe, test_size=0.3, random_state=42)
    # 使用StandardScaler对数据进行标准化
    scaler = StandardScaler()
    X_train_score = scaler.fit_transform(X_train)
    X_test_score = scaler.fit_transform(X_test)

    # 构造各种回归模型
    regressors = [
        GradientBoostingRegressor(random_state=10),
        RandomForestRegressor(random_state=10),
        xgb.XGBRegressor(random_state=10),
        lgb.LGBMRegressor(random_state=10)
    ]

    # 回归模型名称
    regressor_names = [
        'GBDT',
        'Random Forest',
        'XGBoost',
        'LightGBM'
    ]

    # 回归模型参数
    regressor_param_grid = [
        {'GBDT__n_estimators': [50, 100, 200]},
        {'Random Forest__n_estimators': [50, 100, 200]},
        {'XGBoost__n_estimators': [50, 100, 200]},
        {'LightGBM__n_estimators': [50, 100, 200]}
    ]

    # 对具体的回归模型进行GridSearchCV参数调优
    def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y,param_grid):
        response = {}
        # 创建 GridSearchCV 对象，并设置参数
        gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='neg_mean_absolute_error')
        # 在训练集上进行网格搜索
        search = gridsearch.fit(train_x, train_y)
        print("GridSearch最优参数：", search.best_params_)
        # 在测试集上进行预测
        predict_y = gridsearch.predict(test_x)
        # 计算评估指标
        response['predict_y'] = predict_y
        response['mae'] = mean_absolute_error(test_y, predict_y)
        response['explained_variance'] = explained_variance_score(test_y, predict_y)
        response['r2'] = r2_score(test_y, predict_y)
        return response

    mae_scores = []
    explained_variance_scores = []
    r2_scores = []
    # 遍历每个回归模型及其对应的参数网格
    for model, model_name, model_param_grid in zip(regressors, regressor_names, regressor_param_grid):
        pipeline = Pipeline([
            (model_name, model)
        ])
        # 对当前回归模型进行参数调优
        result = GridSearchCV_work(pipeline, X_train_score, y_train, X_test_score, y_test,model_param_grid)
        # 记录评估指标
        mae_scores.append(result['mae'])
        explained_variance_scores.append(result['explained_variance'])
        r2_scores.append(result['r2'])

    # 可视化指标
    x_labels = regressor_names
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # 第一个小图：MAE
    axes[0].bar(x_labels, mae_scores, label='MAE', color='blue')
    axes[0].set_xlabel('Regressor')
    axes[0].set_ylabel('MAE Score')
    axes[0].set_title('MAE Comparison')

    # 第二个小图：Explained Variance
    axes[1].bar(x_labels, explained_variance_scores, label='Explained Variance', color='orange')
    axes[1].set_xlabel('Regressor')
    axes[1].set_ylabel('Explained Variance Score')
    axes[1].set_title('Explained Variance Comparison')

    # 第三个小图：R2 Score
    axes[2].bar(x_labels, r2_scores, label='R2 Score', color='green')
    axes[2].set_xlabel('Regressor')
    axes[2].set_ylabel('R2 Score')
    axes[2].set_title('R2 Score Comparison')

    # 调整子图之间的间距
    plt.tight_layout()
    # 保存图表
    plt.savefig('regression_comparison_separate.png')

    # 找到效果最好的模型
    best_model_index = np.argmax(r2_scores)
    best_model_name = regressor_names[best_model_index]
    best_model = regressors[best_model_index]
    print("最好的回归模型是：", best_model_name)

    # 保存相关指标数据
    result_df = pd.DataFrame({
        'Regressor': regressor_names,
        'MAE': mae_scores,
        'Explained Variance': explained_variance_scores,
        'R2 Score': r2_scores
    })
    result_df.to_csv('regression_metrics.csv', index=False,encoding='utf-8-sig')

    # 保存效果最好的模型
    clf = best_model.fit(X_train_score, y_train)
    predict_y = clf.predict(X_test_score)
    print(predict_y)
    joblib.dump(best_model, 'best_regression_model.pkl')


if __name__ == '__main__':
    pie1()
    bar1()
    bar2()
    pie2()
    line1()
    box_plot()
    pair_plot()
    hea_tmap()
    Modeling_()