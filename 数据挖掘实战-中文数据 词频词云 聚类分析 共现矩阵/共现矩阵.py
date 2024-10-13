import pandas as pd
import numpy as np

def main1(data1,name1):
    documents =[]
    for d in data1['fenci']:
        d1 = str(d).split(" ")
        documents.append(d1)

    # 提取所有词并创建词到索引的映射
    vocab = list(set(word for doc in documents for word in doc))
    vocab_to_index = {word: index for index, word in enumerate(vocab)}

    # 初始化共现矩阵
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)), dtype=int)

    # 填充共现矩阵
    for doc in documents:
        for i in range(len(doc)):
            for j in range(i + 1, len(doc)):
                word1, word2 = doc[i], doc[j]
                index1, index2 = vocab_to_index[word1], vocab_to_index[word2]
                co_occurrence_matrix[index1][index2] += 1
                co_occurrence_matrix[index2][index1] += 1

    # 将矩阵转换为 DataFrame
    co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=vocab, columns=vocab)

    co_occurrence_df.to_excel(f'./{name1}/共现矩阵.xlsx')


if __name__ == '__main__':
    list_type = ['阶段1','阶段2','阶段3']
    list_name3 = ['data1','data2','data3']
    for t, n in zip(list_type, list_name3):
        df = pd.read_csv(f'{n}.csv')
        main1(df,t)

    data1 = pd.read_csv('data1.csv')
    data2 = pd.read_csv('data2.csv')
    data3 = pd.read_csv('data3.csv')
    data4 = pd.concat([data1,data2,data3],axis=0)
    main1(data4,'总内容')