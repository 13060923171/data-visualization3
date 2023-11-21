import pandas as pd
import numpy as np
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
from sklearn.metrics import classification_report
import scipy
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import itertools

#ip处理
def ip_chuli1(x):
    x1 = str(x).split("：")
    return x1[-1]

#ip处理
def ip_chuli2(x):
    provinces = ['安徽', '澳门', '北京', '重庆', '福建', '甘肃', '广东', '广西', '贵州', '海南', '河北', '黑龙江', '河南', '湖北', '湖南', '江苏', '江西',
                 '吉林', '辽宁', '内蒙古', '宁夏', '青海', '山东', '上海', '山西', '陕西', '四川', '台湾', '天津', '西藏', '香港', '新疆', '云南', '浙江']

    if x in provinces:
        #表示属于国内
        return 1
    else:
        #表示属于外国
        return 0


def jianjie_processing(x):
    x1 = str(x)
    if x1 == '暂无简介':
        #表示低活跃用户
        return 0
    else:
        #表示高活跃用户
        return 1

#数字处理
def number_data(x):
    try:
        x1 = str(x)
        number = re.findall(r'\d+', x1)
        return number[0]
    except:
        return 0

#粉丝处理
def fan_chuli(x):
    try:
        x1 = int(x)
        return x1
    except:
        if '万' in x:
            x1 = str(x).replace('万','')
            x1 = float(x1) * 10000
            return x1

#停用词函数
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


data = pd.read_csv('./data/data.csv')
data['ip属地'] = data['ip属地'].apply(ip_chuli1)
data['属地分类'] = data['ip属地'].apply(ip_chuli2)
data['简介分类'] = data['简介'].apply(jianjie_processing)
data['博文数'] = data['博文数'].apply(number_data)
data['粉丝量'] = data['粉丝量'].apply(fan_chuli)
data['关注量'] = data['关注量'].apply(fan_chuli)
data['关注量'] = data['关注量'].astype('int')
data['粉丝量'] = data['粉丝量'].astype('int')
data['博文数'] = data['博文数'].astype('int')
data['属地分类'] = data['属地分类'].astype('int')
data['简介分类'] = data['简介分类'].astype('int')

data1 = data[['属地分类','简介分类','粉丝量','关注量','博文数']]

#将数据进行[0,1]规范化
scaled_x = preprocessing.scale(data1)

list_scaler = []
for m in scaled_x:
    number1 = m[0] + m[1] + m[2] + m[3] + m[4]
    list_scaler.append(number1)

data['影响力'] = list_scaler
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.hist(list(data['影响力']), bins=np.arange(-3, 3, 0.5), edgecolor="black")
plt.title('用户影响力分布图')
plt.savefig('./img/用户影响力分布图.png')
plt.show()


def class_type(x):
    x1 = float(x)
    if x1 <= -2:
        return 0
    elif -2 < x1 <= 0:
        return 1
    elif 0 < x1 <= 1:
        return 2
    else:
        return 3

data['用户分类'] = data['影响力'].apply(class_type)



# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data1, list(data['用户分类']), test_size=0.4, random_state=42)

# 利用网格搜索进行超参数调优
param_grid = {'n_estimators': [50, 100, 200],
              'max_features': ['sqrt', 'log2'],
              'max_depth' : [4, 5, 6, 7, 8],
              'criterion' :['gini', 'entropy']
             }
rfc = RandomForestClassifier(random_state=1)
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = 5)
CV_rfc.fit(X_train, y_train)

# 输出最佳参数
print(CV_rfc.best_params_)

# 使用最佳参数重建模型
rfc1=RandomForestClassifier(random_state=1, n_estimators=100, max_features='sqrt', max_depth=8, criterion='entropy')
rfc1.fit(X_train, y_train)

# 进行预测
pred = rfc1.predict(X_test)

# 输出模型评估结果
print('Accuracy score: ', accuracy_score(y_test,pred))
print('Confusion Matrix: \n', confusion_matrix(y_test,pred))

confusion = confusion_matrix(y_test,pred)
#画混淆矩阵
def plot_confusion_matrix(cm,classes, cmap = plt.cm.Blues):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure()
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title('分类混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=0)
    plt.yticks(tick_marks,classes)

    thresh = cm.max() / 2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment='center',
                 color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('真实值')
    plt.xlabel('预测值')
    plt.savefig('./img/混淆矩阵.png')
    plt.show()


plot_confusion_matrix(confusion,classes=['机器用户','低活跃用户','普通用户','高活跃用户'])
#创建分类报告
clf_report = classification_report(y_test, pred)

# 将分类报告保存为 .txt 文件
with open('./data/RandomForest_report.txt', 'w') as f:
    f.write(clf_report)

data = data.drop(['影响力','属地分类','简介分类'],axis=1)
data.to_csv('./data/new_data.csv',encoding='utf-8-sig',index=False)
# # 将分类报告保存为 .txt 文件
# with open('./data/RandomForest_Confusion_Matrix.txt', 'w') as f:
#     f.write(confusion_matrix(y_test,pred))




