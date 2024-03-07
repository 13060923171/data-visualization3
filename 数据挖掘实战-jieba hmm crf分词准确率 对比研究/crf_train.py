from sklearn_crfsuite import CRF
import re
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def word2features(train, i):
    word = train[i].split(':')[0]
    postag = train[i].split(':')[1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }

    if i > 0:
        word1 = train[i - 1].split(':')[0]
        postag1 = train[i - 1].split(':')[1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(train) - 1:
        word1 = train[i + 1].split(':')[0]
        postag1 = train[i + 1].split(':')[1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def train2features(train):
    return [word2features(train, i) for i in range(len(train))]

# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


clean_dict = {}
with open('文本语料库.txt', 'r', encoding='utf-8-sig') as f:
    corpus = f.readlines()
for line in corpus:
    items = line.split()
    for item in items:
        parts = item.rsplit('/', 1)
        if len(parts) > 1:
            key, value = parts[0], parts[1]
            if key not in clean_dict and is_all_chinese(key) == True:
                clean_dict[key] = value

word_dict = {}
for key, value in clean_dict.items():
    if value not in word_dict.values():
        word_dict[key] = value


train_set = []
for key,values in word_dict.items():
    word = str(key) + ":" + str(values)
    train_set.append(word)
train_set_features = train2features(train_set)

# 标签序列
y_train = [item.split(':')[1] for item in train_set]

# CRF模型初始化
crf = CRF(
    algorithm='lbfgs',
    max_iterations=20,
    all_possible_transitions=True)
crf.fit([train_set_features], [y_train])


def predict2features(test):
    return [{
        'bias': 1.0,
        'word': word,
        'word.lower()': word.lower(),
    } for word in test]

def output_segmentation(test_data, y_pred):
    segmented_sentences = []
    current_segment = ""
    for word, tag in zip(test_data, y_pred):
        if tag.startswith("B"):
            if current_segment:
                segmented_sentences.append(current_segment)
            current_segment = word
        elif tag.startswith("I"):
            current_segment += word
        else:
            if current_segment:
                segmented_sentences.append(current_segment)
                current_segment = ""
            segmented_sentences.append(word)
    if current_segment:
        segmented_sentences.append(current_segment)
    return segmented_sentences


def main1():
    list_true = []
    df = pd.read_csv('data.csv')
    test_data = []
    test_set = []
    for i in df['fenci']:
        word = str(i).split(" ")
        for w in word:
            test_data.append(w)

    for w in test_data:
        true_flag = word_dict.get(w)
        if true_flag is not None:
            list_true.append(true_flag)
            test_set.append(w)

    test_set_features = predict2features(test_set)
    y_pred = crf.predict([test_set_features])
    cr_prec, cr_rec, cr_f_score, _ = precision_recall_fscore_support(list_true, y_pred[0], average='weighted')
    data = pd.DataFrame()
    data['Precision'] = ['{:.2f}'.format(cr_prec)]
    data['Recall'] = ['{:.2f}'.format(cr_rec)]
    data['FScore'] = ['{:.2f}'.format(cr_f_score)]
    data.to_csv('prec_crf.csv',index=False,encoding='utf-8-sig')


def main2(fenci):
    list_true = []
    test_data = []
    test_set = []
    word = str(fenci).split(" ")
    for w in word:
        test_data.append(w)

    for w in test_data:
        true_flag = word_dict.get(w)
        # if true_flag is not None:
        list_true.append(true_flag)
        test_set.append(w)

    test_set_features = predict2features(test_set)
    y_pred = crf.predict([test_set_features])
    segmented_result = output_segmentation(test_data, y_pred[0])
    segmented_result = ' '.join(segmented_result)
    return segmented_result


main1()
df = pd.read_csv('data.csv')
df1 = df
df1['fenci'] = df['fenci'].apply(main2)
df1.to_csv('crf_data.csv',index=False,encoding='utf-8-sig')
