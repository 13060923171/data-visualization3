import pandas as pd
import os
from tqdm import tqdm

def main(os_name,txt_name):
    with open('./待处理文档整理/{}/{}'.format(os_name,txt_name),'r',encoding='gb18030', errors='ignore') as f:
        content = f.readlines()

    # def start():
    #     for c in range(len(content)):
    #         if '文字记录' in content[c]:
    #             return int(c)
    #         else:
    #             return int(0)

    def start():
        for c in range(len(content)):
            if '第1章' in content[c]:
                return int(c)
            else:
                return int(0)

    comment = ''
    number_start = start()
    for c in content[number_start+1:]:
        if '第' in c and '章' in c:
            pass
        # if '关键词' in c:
        #     pass
        else:
            c = str(c).replace('\n','')
            comment += str(c)

    list_data = []
    list_name = []
    list_os = []
    if len(comment)/3000 != int(len(comment)/3000):
        yushu = (int(len(comment)/3000) + 1) * 3000
        yushu1 = [x for x in range(0,yushu+1,3000)]
        for l in range(len(yushu1)-1):
            list_name.append(txt_name)
        for l in range(len(yushu1)-1):
            list_os.append(os_name)
        for i in range(0,len(yushu1)-1):
            list_data.append(comment[yushu1[i]:yushu1[i+1]])
    else:
        yushu = (int(len(comment) / 3000)) * 3000
        yushu1 = [x for x in range(0, yushu + 1, 3000)]
        for l in range(len(yushu1) - 1):
            list_name.append(txt_name)
        for l in range(len(yushu1) - 1):
            list_os.append(os_name)
        for i in range(0, len(yushu1) - 1):
            list_data.append(comment[yushu1[i]:yushu1[i+1]])

    df = pd.DataFrame()
    df['os_name'] = list_os
    df['txt_name'] = list_name
    df['txt_paragraph'] = list_data
    df.to_csv('data1.csv',mode='a+',encoding='gb18030',index=False,header=False)


if __name__ == '__main__':
    df = pd.DataFrame()
    df['os_name'] = ['os_name']
    df['txt_name'] = ['txt_name']
    df['txt_paragraph'] = ['txt_paragraph']
    df.to_csv('data1.csv', mode='w',encoding='gb18030',index=False, header=False)
    # list_or = ['冰糖炖雪梨txt',"程序员那么可爱","你是我的荣耀TXT","你微笑时很美","亲爱的热爱的TXT","甜蜜暴击字幕"]
    list_or = ['小说文本-A', "小说文本-B", "小说文本-C"]
    for l in tqdm(list_or):
        filePath = './待处理文档整理/{}'.format(l)
        for j in tqdm(os.listdir(filePath)):
            main(l,j)
