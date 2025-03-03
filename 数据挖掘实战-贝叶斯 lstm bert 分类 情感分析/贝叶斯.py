import pandas as pd
import jieba
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用Agg后端
sns.set_style(style="whitegrid")


# 读取数据
data1 = pd.read_csv('./train/新_博文表.csv')
data2 = pd.read_csv('./train/新_评论表.csv')

df1 = pd.DataFrame()
df1['fenci'] = data1['fenci']
df1['label'] = data1['label']

df2 = pd.DataFrame()
df2['fenci'] = data2['fenci']
df2['label'] = data2['label']

df = pd.concat([df1,df2],axis=0)

if not os.path.exists("./贝叶斯"):
    os.mkdir("./贝叶斯")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['fenci'], df['label'], test_size=0.3, random_state=42)

# 标签编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# 使用 CountVectorizer
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)


# 评估函数
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 创建指标数据
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [accuracy, precision, recall, f1]

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#ccd5ae', '#e9edc9', '#fefae0', '#faedcd'])
    plt.ylim(0, 1.1)
    plt.title(f'Model Performance Metrics: {model_name}')
    plt.ylabel('Score')

    # 在柱子上方显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.savefig(f'./贝叶斯/{model_name}_性能指标.png')
    plt.close()  # 关闭图形防止内存泄漏

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1

def nb1():
    # 训练 CountVectorizer 特征的贝叶斯模型，使用网格搜索
    nb_count_params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0]}
    nb_count_grid = GridSearchCV(MultinomialNB(), nb_count_params, cv=5, scoring='accuracy')
    nb_count_grid.fit(X_train_count, y_train)
    best_nb_count = nb_count_grid.best_estimator_
    print(best_nb_count)
    y_pred_count = best_nb_count.predict(X_test_count)
    metrics_nb_count = evaluate_model(y_test, y_pred_count, 'Naive Bayes')

    # 预测并保存结果
    def save_predictions(data, filename):
        # 转换特征
        X = count_vectorizer.transform(data['fenci'])
        # 预测数值标签
        y_pred_numeric = best_nb_count.predict(X)
        # 转换回原始标签
        data['情感分类'] = label_encoder.inverse_transform(y_pred_numeric)
        data = data.drop(['label','score'], axis=1)
        # 保存结果
        data.to_excel(f'./贝叶斯/predictions_{filename}', index=False)
        print(f"预测结果已保存至 predictions_{filename}")

    # 处理原始数据
    save_predictions(data1, '新_博文表.xlsx')
    save_predictions(data2, '新_评论表.xlsx')

    def emotion_pie(label):
        d = {}
        for l in label:
            d[l] = d.get(l, 0) + 1
        x_data = []
        y_data = []
        for x,y in d.items():
            x_data.append(x)
            y_data.append(y)
        plt.figure(figsize=(9, 6), dpi=500)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.pie(y_data, labels=x_data, startangle=0, autopct='%1.2f%%')
        plt.title(f'情感分布情况')
        plt.tight_layout()
        # 添加图例
        plt.legend(x_data, loc='lower right')
        plt.savefig(f'./贝叶斯/情感分布情况.png')

    new_df1 = pd.read_excel('./贝叶斯/predictions_新_博文表.xlsx')
    new_df2 = pd.read_excel('./贝叶斯/predictions_新_评论表.xlsx')
    label1 = new_df1['情感分类'].tolist()
    label2 = new_df2['情感分类'].tolist()
    label3 = label1 + label2
    emotion_pie(label3)


    # 计算每个点的FPR和TPR，这里假设正类是第二类（索引为1）
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_count)

    # 计算PR曲线的数据点
    precision, recall, _ = precision_recall_curve(y_test, y_pred_count)

    # 计算AUC值
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('./贝叶斯/NB_ROC曲线.png')
    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig('./贝叶斯/NB_PR曲线.png')


if __name__ == '__main__':
    nb1()

