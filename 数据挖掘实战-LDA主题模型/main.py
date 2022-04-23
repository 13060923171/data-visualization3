import pandas as pd
import re
import jieba
import jieba.analyse
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
df = pd.read_excel('好大夫.xlsx')
content = df['问题']
content = content.drop_duplicates(keep='first')


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def get_cut_words(content_series):
    # 读入停用词表
    stop_words = []

    with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())
    # 分词
    word_num = jieba.lcut(content_series.str.cat(sep='。'), cut_all=False)

    # 条件筛选
    word_num_selected = [i for i in word_num if i not in stop_words and len(i) >= 2 and is_all_chinese(i) == True]
    return word_num_selected



#定义机械压缩函数
def yasuo(st):
    for i in range(1,int(len(st)/2)+1):
        for j in range(len(st)):
            if st[j:j+i] == st[j+i:j+2*i]:
                k = j + i
                while st[k:k+i] == st[k+i:k+2*i] and k<len(st):
                    k = k + i
                st = st[:j] + st[k:]
    return st



content = content.astype(str)
content = content.apply(yasuo)
content = content.dropna(how='any')
text1 = get_cut_words(content_series=content)


counts = {}
for t in text1:
    counts[t] = counts.get(t,0)+1

ls = list(counts.items())
ls.sort(key=lambda x:x[1],reverse=True)
x_data = []
y_data = []

for key,values in ls[:200]:
    x_data.append(key)
    y_data.append(values)

df1 = pd.DataFrame()
df1['word'] = x_data
df1['counts'] = y_data
df1.to_csv('高频词.csv',encoding="utf-8-sig")



jieba.analyse.set_stop_words('stopwords_cn.txt')
f = open('C-class-fenci.txt', 'w', encoding='utf-8')
for line in content:
    line = line.strip('\n')
    # 停用词过滤
    line = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", line)
    seg_list = jieba.cut(line, cut_all=False)
    cut_words = (" ".join(seg_list))

    # 计算关键词
    all_words = cut_words.split()
    c = Counter()
    for x in all_words:
        if len(x) > 1 and x != '\r\n':
            if is_all_chinese(x) == True:
                c[x] += 1
    # Top50
    output = ""
    # print('\n词频统计结果：')
    for (k, v) in c.most_common():
        # print("%s:%d"%(k,v))
        output += k + " "

    f.write(output + "\n")
else:
    f.close()



corpus = []

# 读取预料 一行预料为一个文档
for line in open('C-class-fenci.txt', 'r', encoding='utf-8').readlines():
    corpus.append(line.strip())

# 设置特征数
n_features = 2000

tf_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                max_df=0.99,
                                min_df=0.002)  # 去除文档内出现几率过大或过小的词汇

tf = tf_vectorizer.fit_transform(corpus)

# 设置主题数
n_topics = 5

lda = LatentDirichletAllocation(n_components=n_topics,
                                max_iter=100,
                                learning_method='online',
                                learning_offset=50,
                                random_state=0)
lda.fit(tf)

# 显示主题数 model.topic_word_
print(lda.components_)
# 几个主题就是几行 多少个关键词就是几列
print(lda.components_.shape)

# 计算困惑度
print(u'困惑度：')
print(lda.perplexity(tf,sub_sampling = False))



# 主题-关键词分布
def print_top_words(model, tf_feature_names, n_top_words):
    for topic_idx,topic in enumerate(model.components_):  # lda.component相当于model.topic_word_
        df2 = pd.DataFrame()
        print('Topic #%d:' % topic_idx)
        print(' '.join([tf_feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
        df2['Topic #%d:'] = ['Topic #%d:' % topic_idx]
        df2['word'] = [' '.join([tf_feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]])]
        df2.to_csv('lda结果.csv',encoding='utf-8-sig',mode='a+',header=None)
        print("")

# 定义好函数之后 暂定每个主题输出前20个关键词
n_top_words = 20
tf_feature_names = tf_vectorizer.get_feature_names()
# 调用函数
print_top_words(lda, tf_feature_names, n_top_words)