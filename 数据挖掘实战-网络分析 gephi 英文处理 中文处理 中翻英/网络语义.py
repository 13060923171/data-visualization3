import os
import csv
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def process_csv_file(input_file_path):
    word = []  # 记录关键词
    df = pd.read_excel(input_file_path)

    for line in df['分词1']:
        line = line.strip()
        for n in line.split(' '):
            if n not in word:
                word.append(n)
    print(len(word))  # 打印关键词总数

    word_vector = coo_matrix((len(word), len(word)), dtype=np.int8).toarray()
    print(word_vector.shape)

    for line in df['分词1']:
        line = line.strip()
        nums = line.split(' ')
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                w1, w2 = nums[i], nums[j]
                n1, n2 = word.index(w1), word.index(w2)
                if n1 <= n2:
                    word_vector[n1][n2] += 1
                else:
                    word_vector[n2][n1] += 1
    return word, word_vector


def write_txt_output(output_path, word, word_vector):
    with open(output_path, 'a+', encoding='utf-8') as res:
        for i in range(len(word)):
            for j in range(len(word)):
                if word_vector[i][j] > 0:
                    res.write(f"{word[i]} {word[j]} {int(word_vector[i][j])}\n")


def write_csv_output(output_path, word, word_vector):
    with open(output_path, 'w', encoding='gbk', newline='') as c:
        writer = csv.writer(c)
        writer.writerow(['Word1', 'Word2', 'Weight'])
        for i in range(len(word)):
            for j in range(len(word)):
                if word_vector[i][j] > 0:
                    writer.writerow([word[i], word[j], int(word_vector[i][j])])


def process_weight_csv(input_csv_path, output_entity_path, output_weight_path):
    data = pd.read_csv(input_csv_path, encoding='gbk')
    df = pd.DataFrame({'id': data['Word1'], 'label': data['Word1']}).drop_duplicates().dropna()
    df.to_csv(output_entity_path, encoding='gbk', index=False)

    df1 = data[['Word1', 'Word2', 'Weight']].rename(columns={'Word1': 'Source', 'Word2': 'Target'})
    df1['Type'] = 'Undirected'
    df1['Weight'] = df1['Weight'].astype(int)
    new_df1 = df1[df1['Weight'] >= 20].dropna()
    new_df1.to_csv(output_weight_path, encoding='gbk', index=False)


def main():
    data_dirs = ['./data2-英文/']
    for data_dir in data_dirs:
        input_file_path = os.path.join(data_dir, 'en_comment.xlsx')  # 修改为读取 CSV 文件
        output_txt_path = os.path.join(data_dir, 'weight.txt')
        output_csv_path = os.path.join(data_dir, 'weight.csv')
        output_entity_path = os.path.join(data_dir, 'entity.csv')
        output_weight_path = os.path.join(data_dir, f'{os.path.basename(data_dir)}_weight.csv')

        word, word_vector = process_csv_file(input_file_path)
        write_txt_output(output_txt_path, word, word_vector)
        write_csv_output(output_csv_path, word, word_vector)
        process_weight_csv(output_csv_path, output_entity_path, output_weight_path)


if __name__ == '__main__':
    main()