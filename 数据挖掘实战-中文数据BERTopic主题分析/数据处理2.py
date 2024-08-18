import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
from tqdm import tqdm



list_name1 = ['肌肉紧张','焦躁不安','紧张兴奋','疲劳','睡眠障碍','思维空白','易怒','注意力不集中']
list_dataframe = []
for l in list_name1:
    df = pd.read_excel(f'清洗后-{l}-二次清洗.xlsx')
    df1 = df.drop_duplicates(subset=['博文内容'])
    list_dataframe.append(df1)

data = pd.concat(list_dataframe,axis=0)
data.to_excel('data.xlsx',index=False)

