import hashlib
import json
import os
import pprint
import random
import sys
import time
import uuid

import copyheaders
import pandas
import requests
from lxml import etree

headers = copyheaders.headers_raw_to_dict(b"""
Accept:*/*
Authorization:Bearer ya29.a0AfB_byCPal0ClETv1UfrM4lcrk-j-I78w3CCE8CH98v6qK707MD1dGvHDZGJuCARyjOR40oiD-9gr2wHF8xYyOeVGlOUBcmRiIc7f0cpWkkKAbWkhT9kpfWxnTmp4at5LhMJGXPH7tMVxBaka6lQ2s4LPPtW6DApqQbfQxMaCgYKASUSARISFQGOcNnCEe8K64iNS7PqcHD16JveXg0174
Referer:https://content-youtube.googleapis.com/static/proxy.html?usegapi=1&jsh=m%3B%2F_%2Fscs%2Fabc-static%2F_%2Fjs%2Fk%3Dgapi.lb.zh_CN.4lkP9HfUARs.O%2Fd%3D1%2Frs%3DAHpOoo8MzV9H712hx3UhnN0D-Rtu2UQIRw%2Fm%3D__features__
Sec-Ch-Ua:"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"
Sec-Ch-Ua-Mobile:?0
Sec-Ch-Ua-Platform:"Windows"
Sec-Fetch-Dest:empty
Sec-Fetch-Mode:cors
Sec-Fetch-Site:same-origin
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36
X-Client-Data:CJa2yQEIorbJAQipncoBCJ7zygEIlKHLAQiFoM0BCNuxzQEI3L3NAQi5yM0BCJLKzQEIucrNAQjF1M0BCJjWzQEIp9jNAQji2s0BCPnA1BUYj87NAQ==
X-Clientdetails:appVersion=5.0%20(Windows%20NT%2010.0%3B%20Win64%3B%20x64)%20AppleWebKit%2F537.36%20(KHTML%2C%20like%20Gecko)%20Chrome%2F118.0.0.0%20Safari%2F537.36&platform=Win32&userAgent=Mozilla%2F5.0%20(Windows%20NT%2010.0%3B%20Win64%3B%20x64)%20AppleWebKit%2F537.36%20(KHTML%2C%20like%20Gecko)%20Chrome%2F118.0.0.0%20Safari%2F537.36
X-Goog-Encode-Response-If-Executable:base64
X-Javascript-User-Agent:apix/3.0.0 google-api-javascript-client/1.1.0
X-Origin:https://explorer.apis.google.com
X-Referer:https://explorer.apis.google.com
X-Requested-With:XMLHttpRequest
""")

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

api_key = 'AIzaSyCGfib5EKZ_rsZ4jfVoSQ-A42Qo8C39LyU'

def send_get(url):
    while 1:
        try:
            res = requests.get(
                url,
                timeout=(4,5),
            )
            time.sleep(.1)
            return res.json()
        except Exception as e:
            print(f"some error:{e}")
            time.sleep(1)


  # 随机字符串

def getele(t):
    try:
        return etree.HTML(t).xpath("string(.)")
    except:
        return t


def get_comments(vid,ire):
    print(vid)
    pageconfig = {"count": 0, "index": 0}
    api_uri = 'https://www.googleapis.com/youtube/v3/'
    config_next = {"p": ''}

    for page in range(100):
        pageconfig["index"] += 1
        video_info_search_url = f'{api_uri}commentThreads?part=id,snippet,replies&key={api_key}&videoId={vid}&pageToken={config_next["p"]}&maxResults=100'
        res = send_get(video_info_search_url)
        items = res.get("items")
        if items == None:
            items = []
        for item in items:
            try:
                pageconfig["count"] += 1
                snippet = item.get("snippet").get("topLevelComment").get("snippet")
                saveitem = ire.copy()

                saveitem["评论id"] = item.get("id")
                saveitem["评论点赞数"] = snippet.get('likeCount')
                saveitem["评论内容"] = getele(snippet.get("textDisplay"))
                saveitem["评论发布者"] = snippet.get("authorDisplayName")
                saveitem["评论发布时间"] = snippet.get("publishedAt")
                print(pageconfig, saveitem)


                with open(f"replay_cms.txt", 'a', encoding='utf-8') as ff:
                    ff.write(json.dumps(saveitem))
                    ff.write('\n')


            except Exception as e:
                print(f">>>解析问题： {e}")

        config_next["p"] = res.get("nextPageToken")

        if len(items) <= 90 or config_next["p"] == None or config_next["p"] == '':
            print('break', pageconfig["index"])
            break
        print(f'下一页：{page} {config_next["p"]}')
        time.sleep(0.5)







if  __name__ == "__main__":

    ##采集十万条评论大概需要百度翻译开发者200  块钱左右，采集全确保有这么多

    records = pandas.read_excel("post_detail.xlsx").to_dict(orient='records')[:]
    for ire in records:
        if ire.get("评论数") <= 0:
            continue
        vauth = ire.get("作者")
        vid = ire.get("视频id")
        vtitle = ire.get("标题")
        get_comments(vid=vid,ire=ire)
    with open("replay_cms.txt",'r',encoding='utf-8') as f:
        results =  [json.loads(i.strip()) for i in f.readlines() if i.strip() != '']
    df = pandas.DataFrame(results)
    df.to_excel("comment_list.xlsx",index=False)

