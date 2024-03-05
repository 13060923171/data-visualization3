import jieba
import jieba.posseg as pseg
import pynlpir
import pycrfsuite
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


word_dict = {}
with open('文本语料库.txt','r',encoding='utf-8-sig')as f:
    corpus = f.readlines()
for line in corpus:
    items = line.split()
    for item in items:
        parts = item.rsplit('/', 1)
        if len(parts) > 1:
            key, value = parts[0], parts[1]
            if key not in word_dict:
                word_dict[key] = value

# 导入停用词列表
stop_words = []
with open("stopwords_cn.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())

#去掉标点符号，以及机械压缩
def preprocess_word(word):
    word1 = str(word)
    # word1 = re.sub(r'转发微博', '', word1)
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


# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


# 数据集
list_data = ['菜品比以前差了，餐具档次也降了级，价钱却升了，但总体来说还可以接受。']

# # 使用Jieba分词
jieba_result = [pseg.cut(d) for d in list_data]
jieba_result = [(word.word, word.flag) for sublist in jieba_result for word in sublist]
jieba_result1 = []
list_true = []
for word,flag in jieba_result:
    if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
        true_flag = word_dict.get(word)
        if true_flag is not None:
            list_true.append(true_flag)
            jieba_result1.append(flag)

# 打开你的命令行界面
# 输入 pynlpir update
# 使用NLPIR HMM
pynlpir.open()
hmm_result = [pynlpir.segment(d, pos_tagging=True) for d in list_data]
hmm_result = [(word[0], word[1]) for sublist in hmm_result for word in sublist]
hmm_result1= []
list_true = []
for word,flag in hmm_result:
    if word not in stop_words and len(word) >= 2 and is_all_chinese(word) == True:
        true_flag = word_dict.get(word)
        if true_flag is not None:
            list_true.append(true_flag)
            hmm_result1.append(flag)
pynlpir.close()


# 加载 CRF 模型
tagger = pycrfsuite.Tagger()
tagger.open('crf_model.crfsuite')
# 定义特征函数
def word2features(sent, i):
    word = sent[i][0]

    features = {
        'word': word,
        # 添加更多特征
    }
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]
# 分词和词性标注
crf_result = []
list_true = []
for d in list_data:
    words = pseg.lcut(d)
    crf_tags = tagger.tag(sent2features([(w.word, '') for w in words]))
    crf_result.append(list(zip([w.word for w in words], crf_tags)))

# 输出结果
for result in crf_result:
    print(result)

# 计算并打印precision、recall、f_score，支持
# jr_prec, jr_rec, jr_f_score, _ = precision_recall_fscore_support(jieba_result2, jieba_result1, average='weighted')
# hr_prec, hr_rec, hr_f_score, _ = precision_recall_fscore_support(list_true, hmm_result, average='weighted')
# cr_prec, cr_rec, cr_f_score, _ = precision_recall_fscore_support(list_true, crf_result, average='weighted')

# print("Jieba: Precision: {:.2f}, Recall: {:.2f}, FScore: {:.2f}".format(jr_prec, jr_rec, jr_f_score))
# print("HMM: Precision: {:.2f}, Recall: {:.2f}, FScore: {:.2f}".format(hr_prec, hr_rec, hr_f_score))
# print("CRF: Precision: {:.2f}, Recall: {:.2f}, FScore: {:.2f}".format(cr_prec, cr_rec, cr_f_score))

# # 对比并可视化
# labels = ['Precision', 'Recall', 'F-Score']
# jieba_metrics = [jr_prec, jr_rec, jr_f_score]
# hmm_metrics = [hr_prec, hr_rec, hr_f_score]
# # crf_metrics = [cr_prec, cr_rec, cr_f_score]
#
# x = np.arange(len(labels))
# width = 0.2
#
# fig, ax = plt.subplots()
# rects1 = ax.barh(x - width/2, jieba_metrics, width, label='Jieba')
# rects2 = ax.barh(x + width/2, hmm_metrics, width, label='HMM')
# # rects3 = ax.barh(x + 3*width/2, crf_metrics, width, label='CRF')
#
# ax.set_xlabel('Metrics')
# ax.set_title('Segmentation Metrics by Algorithm')
# ax.set_yticks(x)
# ax.set_yticklabels(labels)
# ax.legend()
#
# fig.tight_layout()
# plt.show()