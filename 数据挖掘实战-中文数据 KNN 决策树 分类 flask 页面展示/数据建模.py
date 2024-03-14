import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯分类器
from sklearn.neighbors import KNeighborsClassifier  # 导入KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#管道
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data.csv')
le = LabelEncoder()
df['MBTI'] = le.fit_transform(df['MBTI'])
# 保存LabelEncoder模型
joblib.dump(le, 'label_encoder_model.pkl')
content = list(df['分词'])
labels = list(df['MBTI'])
# 通过TfidfVectorizer，将文本转换为特征向量
vectorizer = TfidfVectorizer()
content_vectorized = vectorizer.fit_transform(content)
# 保存TfidfVectorizer模型
joblib.dump(vectorizer, 'tfidf_vectorizer_model.pkl')

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(content_vectorized, labels, test_size=0.2, random_state=0)

# 构造各种分类模型
classifiers = [
    DecisionTreeClassifier(),  # 添加决策树分类器
    KNeighborsClassifier(metric='minkowski'),
    # GaussianNB()
]

# 分类模型名称
classifier_names = [
    'decision_tree',  # 定义新的模型名称
    'kneighborsclassifier',
    # 'gaussian_nb'  # 定义新的模型名称
]

# 分类模型参数
classifier_param_grid = [
    {},  # 参数为空字典
    {'kneighborsclassifier__n_neighbors': [4, 6, 8, 10]},
    # {}  # 添加空字典以表示高斯朴素贝叶斯分类器没有需要调整的参数
]


# 对具体的分类模型进行GridSearchCV参数调优
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid):
    response = {}
    # 创建 GridSearchCV 对象，并设置参数
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy')
    # 在训练集上进行网格搜索
    search = gridsearch.fit(train_x, train_y)
    predict_y = gridsearch.predict(test_x)

    print("GridSearch最优参数：", search.best_params_)
    print("GridSearch最优分数： %0.4lf" % search.best_score_)

    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y, predict_y)
    response['precision_score'] = precision_score(test_y, predict_y, average='weighted')
    response['recall_score'] = recall_score(test_y, predict_y, average='macro')
    response['f1_score'] = f1_score(test_y, predict_y, average='weighted')
    return response


accuracy_score1 = []
precision_score1 = []
recall_score1 = []
f1_score1 = []
# 遍历每个分类模型及其对应的参数网格
for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
        (model_name, model)
    ])
    # 对当前分类模型进行参数调优
    result = GridSearchCV_work(pipeline, X_train, y_train, X_test, y_test, model_param_grid)
    # 记录评估指标
    accuracy_score1.append(result['accuracy_score'])
    precision_score1.append(result['precision_score'])
    recall_score1.append(result['recall_score'])
    f1_score1.append(result['f1_score'])


# 可视化指标
x_labels = classifier_names
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 6))

# 第一个小图：accuracy
axes[0].bar(x_labels, accuracy_score1, label='accuracy', color='blue')
axes[0].set_xlabel('classifier')
axes[0].set_ylabel('accuracy score')
axes[0].set_title('accuracy score Comparison')

# 第二个小图：precision_score
axes[1].bar(x_labels, precision_score1, label='precision', color='orange')
axes[1].set_xlabel('classifier')
axes[1].set_ylabel('precision score')
axes[1].set_title('precision score Comparison')

# 第三个小图：recall_score
axes[2].bar(x_labels, recall_score1, label='recall score', color='green')
axes[2].set_xlabel('classifier')
axes[2].set_ylabel('recall score')
axes[2].set_title('recall score Comparison')

# 第四个小图：f1_score
axes[3].bar(x_labels, f1_score1, label='f1 score', color='green')
axes[3].set_xlabel('classifier')
axes[3].set_ylabel('f1 score')
axes[3].set_title('f1 score Comparison')

# 调整子图之间的间距
plt.tight_layout()
# 保存图表
plt.savefig('Model_indicator_comparison.png')

# 找到效果最好的模型
best_model_index = np.argmax(accuracy_score1)
best_model_name = classifier_names[best_model_index]
best_model = classifiers[best_model_index]
print("最好的分类模型是：", best_model_name)

# 保存相关指标数据
result_df = pd.DataFrame({
    'classifier': classifier_names,
    'accuracy_score': accuracy_score1,
    'precision_score': precision_score1,
    'recall_score': recall_score1,
    'f1_score': f1_score1,
})
result_df.to_csv('Model_indicator_comparison.csv', index=False, encoding='utf-8-sig')

# 保存效果最好的模型
clf = best_model.fit(X_train, y_train)
predict_y = clf.predict(X_test)
joblib.dump(best_model, 'best_classifier_model.pkl')