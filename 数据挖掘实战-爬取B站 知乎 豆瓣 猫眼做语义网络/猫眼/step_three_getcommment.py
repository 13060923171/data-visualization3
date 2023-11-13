import json
import os
import pprint
import time

import copyheaders
import pandas
import redis
import requests

headers = copyheaders.headers_raw_to_dict(b"""
Accept:*/*
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36
""")

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

api_key = 'AIzaSyCGfib5EKZ_rsZ4jfVoSQ-A42Qo8C39LyU'


def send_get(url):
    print(f"》》》正在访问：{url}")
    while 1:
        try:
            res = requests.get(
                url,
                timeout=(4, 5),
                headers=headers
            )
            time.sleep(.1)
            return res.json()
        except Exception as e:
            print(f"some error:{e}")
            time.sleep(1)


# 随机字符串


def get_detail(record):
    vname = record.get("电影名")
    vid = record.get("电影id")
    if str(vid)  == "nan":
        return
    for page in range(1,200):
        offset = page *  20
        url = f'https://api.maoyan.com/review/v2/comments.json?movieId={vid}&level=1&type=3&containSelfComment=false&tag=0&ts=0&offset={offset}&limit=20&channelId=70001&' \
              'version=wallet-v4.6.14&uuid=&platform=13&partner=1&riskLevel=71&optimusCode=10'
        response = send_get(url)
        comments = response.get("data",{}).get("comments",[])

        for item in comments:
            saveitem = record.copy()
            saveitem["评论内容"] = item.get("content")
            saveitem["评论id"] = item.get("id")
            saveitem["评论性别"] = item.get("gender")
            saveitem["ip属地"] = item.get("ipLocName")
            saveitem["评论者"] = item.get("nick")
            saveitem["打分"] = item.get("score")
            saveitem["时间"] = time.strftime('%Y-%m-%d',time.localtime(item.get("time")//1000))
            saveitem["有用数"] = item.get("upCount")
            print(config_number["p"],page,saveitem)
            with open("temp_comment.txt",'a',encoding='utf-8') as f:
                f.write(json.dumps(saveitem))
                f.write('\n')

        if len(comments) < 20:
            break


# except Exception as e:
#     print(f"parse error:{e}")
#     time.sleep(1)


if __name__ == "__main__":
    redis_con = redis.Redis(db=2)
    records = pandas.read_excel("电影表.xlsx", dtype=str).to_dict(orient='records')
    config_number = {"p":0}
    for record in records:
        config_number["p"]+=1
        get_detail(record=record)

    with open("temp_comment.txt",'r',encoding='utf-8') as f:
        comments = [json.loads(i.strip()) for i in f.readlines()]
    df = pandas.DataFrame(comments)
    df.to_excel("评论下载.xlsx",index=False)