import os.path
import time
import pandas
import requests



if not os.path.exists("./data"):
    os.makedirs("./data")






def crawl_data(code,startdate,enddate):

    start_samp = time.mktime(time.strptime(startdate,'%Y%m%d'))
    end_samp = time.mktime(time.strptime(enddate,'%Y%m%d'))


    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "cb": "",
        "secid": code,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": "0",
        "end": "20500101",
        "smplmt": "755",
        "lmt": "1000000",
        "_": "1723733407546"
    }
    response = requests.get(url,headers={},params=params).json()

    klines = response.get("data",{}).get("klines",[])
    records = []
    for kl in klines:
        klparams = kl.split(",")
        saveitem = {}
        saveitem["股票"] = code
        saveitem["日期"] = klparams[0]
        saveitem["开盘价"] = klparams[1]
        saveitem["收盘价"] = klparams[2]
        saveitem["最高价"] = klparams[3]
        saveitem["最低价"] = klparams[4]

        current_date_smap = time.mktime(time.strptime(klparams[0],'%Y-%m-%d'))
        if start_samp <= current_date_smap <= end_samp:
            records.append(saveitem)
        print(code,saveitem)
    df = pandas.DataFrame(records)
    df.to_excel(f'./data/股价表.xlsx',index=False)


if __name__ == "__main__":
    ##采集股价
    crawl_data(code='116.01797', startdate='20220601', enddate='20240731')