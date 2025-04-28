import os.path
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
            time.sleep(.8)
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


def crawl_comment_data(blog_id,ikeyword):
    for comment_type in ['0']:
        blog_user_id = ""
        level_1_params = {
            "flow": comment_type,
            "is_reload": "1",
            "id": blog_id,
            "is_show_bulletin": "2",
            "is_mix": "0",
            "count": "10",
            "uid": blog_user_id,
            "fetch_level": "0",
            "locale": "zh-CN"
        }
        for level_1_page in range(1, 11):
            level_1_url = "https://weibo.com/ajax/statuses/buildComments"

            level_1_response = send_get(level_1_url, headers, level_1_params)

            level_1_comment = level_1_response.get("data", [])

            for tag_comment in level_1_comment:

                saveitem = {}
                saveitem["博文id"] = str(blog_id)
                saveitem["关键字"] = str(ikeyword)
                saveitem["评论id"] = str(tag_comment.get("idstr"))

                saveitem["评论内容"] = tag_comment.get("text_raw")
                saveitem["评论时间"] = strify_time(tag_comment.get("created_at"))
                saveitem["评论点赞数"] = tag_comment.get("like_counts")
                saveitem["评论回复数"] = tag_comment.get("total_number")

                saveitem["用户id"] = tag_comment.get("user", {}).get("idstr")
                saveitem["用户名"] = tag_comment.get("user", {}).get("screen_name")
                saveitem["用户ip属地"] = tag_comment.get("user", {}).get("location")
                saveitem["用户标签"] = tag_comment.get("user", {}).get("description")
                saveitem["用户粉丝数"] = tag_comment.get("user", {}).get("followers_count")
                saveitem["用户关注数"] = tag_comment.get("user", {}).get("friends_count")
                saveitem["用户博文数"] = tag_comment.get("user", {}).get("statuses_count")
                saveitem["用户性别"] = tag_comment.get("user", {}).get("gender")
                saveitem["用户认证信息"] = tag_comment.get("user", {}).get("verified")
                saveitem["用户vip信息"] = tag_comment.get("user", {}).get("svip")
                saveitem["评论类型"] = '一级评论'

                print(blog_id, level_1_page, saveitem)

                try:
                    save_comment(saveitem)

                except Exception as e:
                    print(f'save error:{e}')

                if tag_comment.get("total_number") > 0:
                    level_2_params = {
                        "is_reload": "1",
                        "id": tag_comment.get("idstr"),
                        "is_show_bulletin": "2",
                        "is_mix": "1",
                        "fetch_level": "1",
                        "max_id": "0",
                        "count": "20",
                        "uid": blog_user_id,
                        "locale": "zh-CN"
                    }
                    for level_2_page in range(1, 500):
                        level_2_url = "https://weibo.com/ajax/statuses/buildComments"

                        level_2_response = send_get(level_2_url, headers, level_2_params)
                        level_2_comment = level_2_response.get("data", [])

                        for sub_comment in level_2_comment:
                            saveitem = {}
                            saveitem["博文id"] = str(blog_id)
                            saveitem["关键字"] = str(ikeyword)
                            saveitem["评论id"] = str(sub_comment.get("idstr"))

                            saveitem["评论内容"] = sub_comment.get("text_raw")
                            saveitem["评论时间"] = strify_time(sub_comment.get("created_at"))
                            saveitem["评论点赞数"] = sub_comment.get("like_counts")
                            saveitem["评论回复数"] = sub_comment.get("total_number")

                            saveitem["用户id"] = sub_comment.get("user", {}).get("idstr")
                            saveitem["用户名"] = sub_comment.get("user", {}).get("screen_name")
                            saveitem["用户ip属地"] = sub_comment.get("user", {}).get("location")
                            saveitem["用户标签"] = sub_comment.get("user", {}).get("description")
                            saveitem["用户粉丝数"] = sub_comment.get("user", {}).get("followers_count")
                            saveitem["用户关注数"] = sub_comment.get("user", {}).get("friends_count")
                            saveitem["用户博文数"] = sub_comment.get("user", {}).get("statuses_count")
                            saveitem["用户性别"] = sub_comment.get("user", {}).get("gender")
                            saveitem["用户认证信息"] = sub_comment.get("user", {}).get("verified")
                            saveitem["用户vip信息"] = sub_comment.get("user", {}).get("svip")
                            saveitem["评论类型"] = '二级评论'
                            saveitem["父评论"] = str(tag_comment.get("idstr"))

                            print(blog_id, level_1_page, level_2_page, saveitem)

                            try:
                                save_comment(saveitem)
                            except Exception as e:
                                print(f'save error:{e}')

                        if level_2_response.get("ok") != 1:
                            break
                        if len(str(level_2_response.get("max_id"))) < 10:
                            break
                        if len(level_2_comment) < 10:
                            break

                        level_2_params["max_id"] = level_2_response.get("max_id")

            if level_1_response.get("ok") != 1:
                break
            if len(str(level_1_response.get("max_id"))) < 10:
                break
            if len(level_1_comment) < 10:
                break
            level_1_params["max_id"] = level_1_response.get("max_id")


def save_comment(saveitem):
    with open("./comment.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(saveitem, ensure_ascii=False) + "\n")

if __name__ == '__main__':

    ##登录失效就从浏览器重新复制
    headers["cookie"] = 'SCF=Au11V-R8mBJfrVL-ElAuyRjhTOSM6PNwutXqpN2KwMUWuI9ZSQJG7aL3Q8b_OWCTKnfRWB2wIltXQC37u1erTls.; SINAGLOBAL=1553660159760.0042.1739524809891; UOR=,,tophub.today; ULV=1745080898273:78:17:1:6549893375692.823.1745080898266:1744684468325; ALF=1747810368; SUB=_2A25FAZsQDeRhGeBJ4lUY8y7EyjWIHXVmfpLYrDV8PUJbkNANLVnBkW1NRjn6U42WAlC6Jek3ORU-YLpwkZP9V2eo; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5n9zJq2EFVW1sfyACKT7rj5JpX5K-hUgL.FoqN1KM4e05ReK.2dJLoIp7LxKML1KBLBKnLxKqL1hnLBoMcS0.N1Ke71h24; XSRF-TOKEN=f5RsqnYts8BDAcoGY5SMs_Sf; PC_TOKEN=89ae7f541b; WBStorage=036f61cc|undefined; WBPSESS=S8MbxEPPgMhJABbt7NPbA2Npqq-HVmdZxa3BBs6L-oe6tsqMLeicPhMYoMkQkp3Jy1r26vqfLd4KtZXUcYcRzlNhrZl_FMhu9H0UuUteDHIjZumXO14NKOLg4f_DYVC6VQ0f2F1UoHxUFQao2COmPg=='


    for blog_info in pandas.read_excel("博文表.xlsx").to_dict(orient='records')[:]:
        if blog_info.get("博文评论数") <= 0:
            continue
        try:
            crawl_comment_data(
                blog_id=blog_info.get("_id"),
                ikeyword=blog_info.get("关键字")
            )
        except Exception as e:
            print(f'blog_comment_crawl error:{e}')

    with open("./comment.txt",'r',encoding='utf-8') as f:
        all_comments_search = [json.loads(i.strip()) for i in f.readlines()]

    df = pandas.DataFrame(all_comments_search)
    df.drop_duplicates(inplace=True)
    with pandas.ExcelWriter('评论表.xlsx', engine='xlsxwriter', options={'strings_to_urls': False}) as writer:
        df.to_excel(writer, index=False)
    os.remove("./comment.txt")

