from googletrans import Translator
import pandas as pd
import numpy as np
from tqdm import tqdm
import time


def translate_chinese_to_english(text, retries=3, delay=1):
    translator = Translator()
    for i in range(retries):
        try:
            translation = translator.translate(text, src='zh-CN', dest='en')
            return translation.text
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                print(f"Translation failed after {retries} attempts: {e}")
                return ""


df = pd.read_excel('new_data.xlsx')
df2 = pd.DataFrame()
df2['评论用户id'] = ['评论用户id']
df2['评论id'] = ['评论id']
df2['分词'] = ['分词']
df2['分词-英文'] = ['分词-英文']
df2.to_csv('new_data_trans.csv',mode='w',encoding='utf-8-sig',index=False,header=False)

for d1,d2,d3 in tqdm(zip(df['评论用户id'],df['评论id'],df['分词'])):
    try:
        text = translate_chinese_to_english(d3)
        df2['评论用户id'] = [d1]
        df2['评论id'] = [d2]
        df2['分词'] = [d3]
        df2['分词-英文'] = [text]
        df2.to_csv('new_data_trans.csv',mode='a+',encoding='utf-8-sig',index=False,header=False)
        time.sleep(0.5)
    except Exception as e:
        print(f"Error processing row ({d1}, {d2}, "
              f""
              f""
              f"{d3}): {e}")

print("Translation completed.")