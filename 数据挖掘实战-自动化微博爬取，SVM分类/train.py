import pandas as pd
import numpy as np
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import scipy
from joblib import dump, load
from sklearn.decomposition import PCA

df1 = pd.read_excel('./train/train.xlsx',sheet_name='正常数据')
df2 = pd.read_excel('./train/train.xlsx',sheet_name='简介为空或关粉比大于3')

df1 = df1.drop(['uid','性别','ip属地'],axis=1)
df2 = df2.drop(['uid','性别','ip属地'],axis=1)

#1代表真实用户，0代表机器人
df1['用户特征'] = 1
df2['用户特征'] = 0

data = pd.concat([df1,df2],axis=0)
data = data.drop_duplicates(subset=['昵称'])


def ip_data(x):
    x1 = str(x).split(" ")
    return x1[0]


def ip_chuli(x):
    x1 = str(x).split("：")
    return x1[-1]


def number_data(x):
    try:
        x1 = str(x)
        number = re.findall(r'\d+', x1)
        return number[0]
    except:
        return 0


def fan_chuli(x):
    try:
        x1 = int(x)
        return x1
    except:
        if '万' in x:
            x1 = str(x).replace('万','')
            x1 = float(x1) * 10000
            return x1


stop_words = []
with open("./data/stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


#去掉标点符号，以及机械压缩
def preprocess_word(word):
    word1 = str(word)
    word1 = re.sub(r'转发微博', '', word1)
    word1 = re.sub(r'#\w+#', '', word1)
    word1 = re.sub(r'【.*?】', '', word1)
    word1 = re.sub(r'@[\w]+', '', word1)
    word1 = re.sub(r'[a-zA-Z]', '', word1)
    word1 = re.sub(r'\.\d+', '', word1)
    return word1


def emjio_tihuan(x):
    x1 = str(x)
    x2 = re.sub('(\[.*?\])', "", x1)
    x3 = re.sub(r'@[\w\u2E80-\u9FFF]+:?|\[\w+\]', '', x2)
    x4 = re.sub(r'\n', '', x3)
    return x4


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


# 定义机械压缩函数
def yasuo(st):
    for i in range(1, int(len(st) / 2) + 1):
        for j in range(len(st)):
            if st[j:j + i] == st[j + i:j + 2 * i]:
                k = j + i
                while st[k:k + i] == st[k + i:k + 2 * i] and k < len(st):
                    k = k + i
                st = st[:j] + st[k:]
    return st


def get_cut_words(content_series):
    # 读入停用词表
    # 分词
    word_num = jieba.lcut(content_series, cut_all=False)

    # 条件筛选
    word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]

    return ' '.join(word_num_selected)


def null_paichu(x):
    x1 = str(x)
    if len(x1) != 0:
        return x1
    else:
        return np.NAN


data['简介'] = data['简介'].apply(preprocess_word)
data['简介'] = data['简介'].apply(emjio_tihuan)
data['简介'] = data['简介'].apply(yasuo)
data['简介'] = data['简介'].apply(get_cut_words)
data['简介'] = data['简介'].apply(null_paichu)
data['昵称'] = data['昵称'].apply(preprocess_word)
data['昵称'] = data['昵称'].apply(emjio_tihuan)
data['昵称'] = data['昵称'].apply(yasuo)
data['昵称'] = data['昵称'].apply(get_cut_words)
data['昵称'] = data['昵称'].apply(null_paichu)
data = data.dropna(subset=['简介','昵称'],axis=0)
# 选取标签列
y = data['用户特征']



vectorizer = TfidfVectorizer()
#该类会统计每个词语的tf-idf权值
word1 = vectorizer.fit_transform(list(data['昵称']))
word2 = vectorizer.transform(list(data['简介']))
data1 = data[['粉丝量', '关注量', '博文数']]
# 创建归一化的实例
scaler = MinMaxScaler()
# 在数据集上进行归一化训练
scaler.fit(data1)
# 应用归一化转换
normalized_data = scaler.transform(data1)
x_num_features_sparse = scipy.sparse.csr_matrix(normalized_data)
#汇总所有特征
X = hstack([word1, word2,x_num_features_sparse])

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(normalized_data, y, test_size=0.2, random_state=42)

# 训练SVM模型
clf = svm.SVC()
clf.fit(X_train, y_train)

# 进行预测
y_pred = clf.predict(X_test)

#创建分类报告
clf_report = classification_report(y_test, y_pred)

# 将分类报告保存为 .txt 文件
with open('./data/classification_report.txt', 'w') as f:
    f.write(clf_report)

dump(clf, './model/svm_model.joblib')
dump(scaler, './model/scaler_model.joblib')

