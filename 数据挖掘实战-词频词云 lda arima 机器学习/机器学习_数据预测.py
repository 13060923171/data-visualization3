import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端
import seaborn as sns

df1 = pd.read_csv('./2024Q2/lda_data.csv')
df2 = pd.read_csv('./2024Q3/lda_data.csv')
df3 = pd.read_csv('./2024Q4/lda_data.csv')
df4 = pd.read_csv('./2025Q1/lda_data.csv')
df5 = pd.read_csv('./2025Q2/lda_data.csv')

df = pd.concat([df1,df2,df3,df4,df5],axis=0)


df['发布时间'] = pd.to_datetime(df['发布时间'])
df['day'] = df['发布时间'].dt.to_period('D')
df = df.sort_values(by=['day'],ascending=True)
df = df.dropna(subset=['day'],axis=0)


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
df['收藏数'] = df['收藏数'].apply(data_process)
df['分享数'] = df['分享数'].apply(data_process)
df['关注'] = df['关注'].apply(data_process)
df['粉丝'] = df['粉丝'].apply(data_process)
df['获赞与收藏'] = df['获赞与收藏'].apply(data_process)
# 标签编码
label_encoder = LabelEncoder()
df['类型'] = label_encoder.fit_transform(df['类型'])
df['关键词'] = label_encoder.fit_transform(df['关键词'])

df = df[['关键词','标题字数','图片数量','类型','点赞数','评论数','收藏数','分享数','字数','关注','粉丝','获赞与收藏','fenci']]
df = df.dropna(subset=['点赞数'],axis=0)



# 定义特征列
text_features = ['fenci']
numeric_features = ['关键词','标题字数', '图片数量', '类型', '评论数', '收藏数', '分享数', '字数', '关注', '粉丝', '获赞与收藏']
target = '点赞数'

X = df[text_features + numeric_features]
y = df[target]

# 2. 特征工程
preprocessor = ColumnTransformer(
    transformers=[
        ('text1', TfidfVectorizer(max_features=100, tokenizer=lambda x: x.split()), 'fenci'),
        ('num', StandardScaler(), numeric_features)
    ])

# 3. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 模型构建与调参
models = {
    'RandomForest': {
        'pipe': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor())
        ]),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10]
        }
    },
    'RidgeRegression': {
        'pipe': Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge())
        ]),
        'params': {
            'model__alpha': [0.1, 1.0]
        }
    }
}

results = []
for model_name, config in models.items():
    grid = GridSearchCV(config['pipe'], config['params'], cv=3,
                       scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    # 评估
    y_pred = grid.predict(X_test)
    metrics = {
        'Model': model_name,
        'Best Params': grid.best_params_,
        'R2': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    results.append(metrics)

# 5. 结果保存
results_df = pd.DataFrame(results)
results_df.to_excel('model_performance.xlsx', index=False)

plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
# 6. 可视化
plt.figure(figsize=(15, 5))

# 模型性能对比
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='R2', data=results_df)
plt.title('Model R-squared Comparison')

# 特征相关性（取前10个数值特征）
numeric_df = df[numeric_features + [target]]
corr_matrix = numeric_df.corr()
plt.subplot(1, 2, 2)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Numeric Features Correlation')
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.show()

