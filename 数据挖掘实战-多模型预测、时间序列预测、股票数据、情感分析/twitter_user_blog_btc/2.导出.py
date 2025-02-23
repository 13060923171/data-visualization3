import datetime  ##时间格式化
import json  ##json解析
import os  ##文件路径操作
import random  ##获取随机数
import re  ##正则解析
import time  ##时间格式化
import pandas  ##数据整理导出
import pymongo
import requests  ##发送网络请求

requests.packages.urllib3.disable_warnings()





def parse_cookie():
    with open("./config/newcookie.txt", 'r', encoding='utf-8') as f:
        lines = [i.strip() for i in f.readlines()]
        for line in lines:

            if len([i for i in line.split("|") if 'ct0' in i]) >= 0:

                try:
                    cookies.append([i.strip() for i in line.split("|") if 'ct0' in i][0])
                    print(f"parse cookie:", [i.strip() for i in line.split("|") if 'ct0' in i][0])
                except:
                    pass

def parse_cookie2():
    with open("./config/newcookie2.txt", 'r', encoding='utf-8') as f:
        lines = [i.strip() for i in f.readlines()]
        for line in lines:

                try:

                    sub_lines = line.split(";")
                    auth_token = sub_lines[3]
                    ct0 = sub_lines[4]
                    cookiestr = f'auth_token={auth_token};lang=en;ct0={ct0};'
                    print(cookiestr)
                    cookies.append(cookiestr)

                except:
                    pass


if __name__ == "__main__":
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    cookies = []
    parse_cookie()

    start_date = '2006-01-01'
    end_date = '2025-02-15'

    pymongo_con = pymongo.MongoClient("mongodb://localhost:27017/")
    pymongo_cur = pymongo_con["twitter-btc"]['data']




    df = pandas.DataFrame([i for i in pymongo_cur.find({})])
    df.drop_duplicates(['博文id'], inplace=True)
    with pandas.ExcelWriter('post.xlsx', engine='xlsxwriter', options={'strings_to_urls': False}) as writer:
        df.to_excel(writer, index=False)
