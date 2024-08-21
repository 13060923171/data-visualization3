import urllib.parse

import copyheaders
import pandas
import pymongo
import redis
import requests
import re
import json
import time
import random
from datetime import datetime
import datetime as dtime
import csv
from lxml import etree
from pathlib import Path

headers = {
    "Cookie": 'SINAGLOBAL=1872946714867.0852.1717913645879; SCF=AqKWdiE3VObm6S9YvufIOMJI8IXWkvrIULdPctXrMzEmqqr1Ljun1QeBGeRgq3hlJaxAf09LuFQaAYy-TEVVskE.; UOR=,,cn.bing.com; ALF=1726299650; SUB=_2A25LucFSDeRhGeBJ4lUY8y7EyjWIHXVot1yarDV8PUJbkNANLXLHkW1NRjn6U5fX8VNag2kvflCgNCfOrBzMSMgI; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5n9zJq2EFVW1sfyACKT7rj5JpX5KMhUgL.FoqN1KM4e05ReK.2dJLoIp7LxKML1KBLBKnLxKqL1hnLBoMcS0.N1Ke71h24; _s_tentry=weibo.com; Apache=2398116145061.8794.1723733844091; ULV=1723733844143:46:12:7:2398116145061.8794.1723733844091:1723707654885',
    "Accept": 'application/json, text/plain, */*',
    "Referer": 'https://weibo.com/u/1735618041',
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
}


def send_get(api, headers, params):
    print(f'>>>访问：{api}')
    while 1:
        try:
            res = requests.get(
                api,
                headers=headers,
                timeout=(4, 5),
                params=params,

            )
            if res.status_code != 200:
                print(res.text)
                time.sleep(10)
                continue
            time.sleep(.3)
            return res.json()
        except Exception as e:
            print(f'some error:{e}')
            time.sleep(1)


def strify_time(dd):
    GMT_FORMAT = '%a %b %d %H:%M:%S +0800 %Y'
    timeArray = datetime.strptime(dd, GMT_FORMAT)
    return timeArray.strftime("%Y-%m-%d %H:%M:%S")


##解析html字符串中的文字
def parse_html(text):
    try:
        return etree.HTML(text).xpath("string(.)")
    except:
        return text



def crawl_comment_data(blog):
    blog_user_id = blog.get("用户id")
    blog_id = blog.get("_id")
    level_1_params = {
        "flow": "0",
        "is_reload": "1",
        "id": blog_id,
        "is_show_bulletin": "2",
        "is_mix": "0",
        "count": "10",
        "uid": blog_user_id,
        "fetch_level": "0",
        "locale": "zh-CN"
    }
    for level_1_page in range(1, 100):
        level_1_url = "https://weibo.com/ajax/statuses/buildComments"


        level_1_response = send_get(level_1_url, headers, level_1_params)

        level_1_comment = level_1_response.get("data", [])

        for tag_comment in level_1_comment:

            saveitem = {}
            saveitem["_id"] = tag_comment.get("idstr")
            saveitem["博文id"] = blog_id
            saveitem["评论内容"] = tag_comment.get("text_raw")
            saveitem["评论时间"] = strify_time(tag_comment.get("created_at"))
            saveitem["评论点赞数"] = tag_comment.get("like_counts")
            saveitem["评论回复数"] = tag_comment.get("total_number")
            saveitem["评论类型"] = blog.get("博文类型")

            saveitem["用户id"] = tag_comment.get("user",{}).get("idstr")
            saveitem["用户名"] = tag_comment.get("user",{}).get("screen_name")
            saveitem["用户ip属地"] = tag_comment.get("user",{}).get("location")
            saveitem["用户标签"] = tag_comment.get("user",{}).get("description")
            saveitem["用户粉丝数"] = tag_comment.get("user",{}).get("followers_count")
            saveitem["用户关注数"] = tag_comment.get("user",{}).get("friends_count")
            saveitem["用户博文数"] = tag_comment.get("user",{}).get("statuses_count")
            saveitem["用户性别"] = tag_comment.get("user",{}).get("gender")
            saveitem["用户认证信息"] = tag_comment.get("user",{}).get("verified")
            saveitem["用户vip信息"] = tag_comment.get("user",{}).get("svip")

            print(blog_id, level_1_page, saveitem)

            try:
                all_comment_finds.append(saveitem)
            except Exception as e:
                print(f'save error:{e}')




        if level_1_response.get("ok") != 1:
            break
        if len(str(level_1_response.get("max_id"))) < 10:
            break
        if len(level_1_comment) < 10:
            break
        level_1_params["max_id"] = level_1_response.get("max_id")




if __name__ == '__main__':

        all_comment_finds = []
        all_blog_info = [i for i in pandas.read_excel("./data/博文表.xlsx",dtype=str).to_dict(orient='records')]

        for blog in all_blog_info:


            try:
                crawl_comment_data(blog)
            except Exception as e:
                print(f'blog_comment_crawl error:{e}')

        df = pandas.DataFrame(all_comment_finds)
        writer = pandas.ExcelWriter(f'./data/评论表.xlsx', engine='xlsxwriter', options={'strings_to_urls': False})
        df.to_excel(writer, index=False)
        writer.close()
