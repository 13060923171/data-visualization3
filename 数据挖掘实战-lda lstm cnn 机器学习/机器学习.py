import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 数据准备保持不变
data = pd.read_csv('./lda/lda_data.csv')
X_train, X_test, y_train, y_test = train_test_split(data['fenci'], data['主题类型'], test_size=0.2, random_state=42)

# TF-IDF特征提取
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 模型配置
models = {
    'SVM': (SVC(probability=True), {
        'C': [15],
        'gamma': [0.1],
        'kernel': ['rbf'],
        'degree': [2]
    }),
    'LR': (LogisticRegression(), {
        'solver': ['newton-cg'],
        'multi_class': ['ovr'],
        'class_weight': ['balanced'],
        'max_iter': [100]
    }),
    'LightGBM': (LGBMClassifier(), {
        'boosting_type': ['gbdt'],
        'num_leaves': [10],
        'learning_rate': [0.1],
        'n_estimators': [500]
    })
}

# 评估指标存储
metrics_data = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1': []
}

# 模型训练流程优化
for name, (model, params) in models.items():
    searcher = RandomizedSearchCV(model, params, n_iter=1, cv=3, scoring='accuracy')
    searcher.fit(X_train_tfidf, y_train)

    best_model = searcher.best_estimator_
    y_pred = best_model.predict(X_test_tfidf)

    # 指标计算（添加加权平均处理）
    metrics_data['Model'].append(name)
    metrics_data['Accuracy'].append(accuracy_score(y_test, y_pred))
    metrics_data['Precision'].append(precision_score(y_test, y_pred, average='weighted'))
    metrics_data['Recall'].append(recall_score(y_test, y_pred, average='weighted'))
    metrics_data['F1'].append(f1_score(y_test, y_pred, average='weighted'))

# 可视化设置
plt.figure(figsize=(12, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

# 柱状图参数设置
bar_width = 0.2
index = np.arange(len(models))
colors = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759']  # 专业配色方案

# 绘制各指标柱状图
for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1']):
    values = [metrics_data[metric][j] for j in range(len(models))]
    plt.bar(index + i*bar_width,
            values,
            width=bar_width,
            color=colors[i],
            label=metric)

# 图表装饰
plt.title('模型性能对比', fontsize=14, pad=20)
plt.xlabel('机器学习模型', fontsize=12)
plt.ylabel('得分', fontsize=12)
plt.xticks(index + 1.5*bar_width, metrics_data['Model'])
plt.ylim(0, 1.05)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for i, model in enumerate(metrics_data['Model']):
    for j, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1']):
        value = metrics_data[metric][i]
        plt.text(index[i] + j*bar_width,
                 value + 0.02,
                 f'{value:.4f}',
                 ha='center',
                 fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()

# 保存结果（添加格式化输出）
results_df = pd.DataFrame(metrics_data)
results_df.set_index('Model', inplace=True)
results_df.to_excel('model_metrics.xlsx', float_format="%.4f")