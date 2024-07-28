from googletrans import Translator
import pandas as pd
import numpy as np
from tqdm import tqdm
import time


def translate_chinese_to_english(text, retries=3, delay=1):
    translator = Translator()
    for i in range(retries):
        try:
            translation = translator.translate(text, src='auto', dest='zh-CN')
            return translation.text
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                print(f"Translation failed after {retries} attempts: {e}")
                return ""


df = pd.read_excel('data.xlsx').iloc[2288:]

df2 = pd.DataFrame()
df2['评论ID'] = ['评论ID']
df2['评论文本'] = ['评论文本']
df2['翻译-评论文本'] = ['翻译-评论文本']
df2.to_csv('zh_data1.csv',mode='w',encoding='utf-8-sig',index=False,header=False)

for d1,d2 in tqdm(zip(df['评论ID'],df['评论文本'])):
    try:
        text = translate_chinese_to_english(d2)
        df2['评论ID'] = [d1]
        df2['评论文本'] = [d2]
        df2['翻译-评论文本'] = [text]
        df2.to_csv('zh_data1.csv',mode='a+',encoding='utf-8-sig',index=False,header=False)
        time.sleep(0.5)
    except Exception as e:
        print(f"Error processing row ({d1}, {d2}, "
              f""
              f""
              f""
              f"{d2}): {e}")

print("Translation completed.")