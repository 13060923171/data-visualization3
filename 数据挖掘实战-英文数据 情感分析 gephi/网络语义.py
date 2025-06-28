import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import dok_matrix  # 使用高效的稀疏矩阵格式


def process_csv_file(input_file_path):
    # 创建词汇表和字典映射
    word_set = set()
    df = pd.read_csv(input_file_path)

    # 一次性收集所有词汇
    for tokens in df['fenci'].str.split(' '):
        word_set.update(tokens)

    word_list = sorted(word_set)  # 排序确保索引一致
    word_to_idx = {word: idx for idx, word in enumerate(word_list)}
    print(f"关键词总数: {len(word_list)}")

    # 使用DOK稀疏矩阵格式 (高效增量更新)
    matrix_size = len(word_list)
    coo_dict = defaultdict(int)  # 使用字典记录共现次数

    # 遍历所有微博内容
    for tokens in df['fenci'].str.split(' '):
        tokens = [t for t in tokens if t]  # 移除空字符串

        # 记录每对组合
        for i in range(len(tokens)):
            idx_i = word_to_idx[tokens[i]]
            for j in range(i + 1, len(tokens)):
                idx_j = word_to_idx[tokens[j]]
                # 统一排序索引
                idx_min, idx_max = min(idx_i, idx_j), max(idx_i, idx_j)
                coo_dict[(idx_min, idx_max)] += 1

    return word_list, coo_dict


def write_txt_output(output_path, word_list, coo_dict):
    with open(output_path, 'w', encoding='utf-8-sig') as res:
        for (i, j), count in coo_dict.items():
            res.write(f"{word_list[i]} {word_list[j]} {count}\n")


def write_csv_output(output_path, word_list, coo_dict):
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as c:
        writer = csv.writer(c)
        writer.writerow(['Word1', 'Word2', 'Weight'])
        for (i, j), count in coo_dict.items():
            writer.writerow([word_list[i], word_list[j], count])


# 以下函数保持不变
def process_weight_csv(input_csv_path, output_entity_path, output_weight_path):
    data = pd.read_csv(input_csv_path, encoding='utf-8-sig')
    df = pd.DataFrame({'id': data['Word1'], 'label': data['Word1']}).drop_duplicates().dropna()
    df.to_csv(output_entity_path, encoding='utf-8-sig', index=False)

    df1 = data[['Word1', 'Word2', 'Weight']].rename(columns={'Word1': 'Source', 'Word2': 'Target'})
    df1['Type'] = 'Undirected'
    df1['Weight'] = df1['Weight'].astype(int)
    new_df1 = df1[df1['Weight'] >= 2].dropna()
    new_df1.to_csv(output_weight_path, encoding='utf-8-sig', index=False)


def main():
    data_dirs = ['.']
    for data_dir in data_dirs:
        input_file_path = os.path.join(data_dir, 'new_comment.csv')
        output_txt_path = os.path.join(data_dir, 'weight.txt')
        output_csv_path = os.path.join(data_dir, 'weight.csv')
        output_entity_path = os.path.join(data_dir, 'entity.csv')
        output_weight_path = os.path.join(data_dir, f'new_weight.csv')

        word_list, coo_dict = process_csv_file(input_file_path)
        write_txt_output(output_txt_path, word_list, coo_dict)
        write_csv_output(output_csv_path, word_list, coo_dict)
        process_weight_csv(output_csv_path, output_entity_path, output_weight_path)


if __name__ == '__main__':
    main()