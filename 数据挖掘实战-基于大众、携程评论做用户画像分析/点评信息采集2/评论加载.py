import json
import os

import pandas


def excuteerr(i):
    try:
        return json.loads(i)
    except:
        return {}

sigh_map = {
    'Etc5cFcMSW3noAWh':'布达拉宫',
    'H3HQPtbXLYypOdnw':'大昭寺'
}



new_datas = []
for fileindex,file in enumerate(os.listdir("./config")):
    if '_data.txt' in file:
        with open(f"./config/{file}",'r',encoding='utf-8') as f:
            lines = [excuteerr(i.strip()) for  i in f.readlines()]
        for newl in lines:
            newl["景点"] = sigh_map.get(newl.get("店铺id"))
            new_datas.append(newl)
        print(fileindex,file,len(new_datas))

pandas.DataFrame(new_datas).to_excel("最终数据.xlsx",index=False)