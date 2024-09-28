import pandas as pd

def capitalize_each_word(text):
    words = text.split()
    capitalized_words = [word.capitalize() for word in words]
    return ' '.join(capitalized_words)

df = pd.read_csv('./data2-英文/_weight.csv')
df['Source'] = df['Source'].apply(capitalize_each_word)
df['Target'] = df['Target'].apply(capitalize_each_word)
df.to_csv('./data2-英文/_weight.csv',encoding='utf-8-sig',index=False)