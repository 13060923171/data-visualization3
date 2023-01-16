import emoji
#读取docx中的文本代码示例
import docx
import pandas as pd
file=docx.Document("emoji表情包.docx")
data = {}
for para in file.paragraphs:
    # print(para.text)
    a = emoji.demojize(para.text)
    b = str(a).split(':')
    for c in b:
        if len(c) != 0:
            d = ":" + c + ":"
            data[d] = data.get(d,0)+1

ls = list(data.items())
ls.sort(key=lambda x:x[1],reverse=True)

x_data = []
y_data = []
for key,values in ls:
    x_data.append(emoji.emojize(key))
    y_data.append(values)

df = pd.DataFrame()
df['emoji'] = x_data
df['counts'] = y_data
df.to_excel('emoji.xlsx',encoding='Base16',index=False)
