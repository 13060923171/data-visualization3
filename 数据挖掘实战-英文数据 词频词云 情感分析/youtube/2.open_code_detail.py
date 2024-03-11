import hashlib
import json
import os
import pprint
import random
import re
import time

import copyheaders
import pandas
import requests

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
htmlheaders = copyheaders.headers_raw_to_dict(b"""
accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
accept-language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6
cache-control: no-cache
cookie: VISITOR_INFO1_LIVE=pzNaW-5TpDs; PREF=tz=Asia.Shanghai; YSC=eI3HWxy5CYk; GPS=1
pragma: no-cache
sec-ch-ua: " Not A;Brand";v="99", "Chromium";v="98", "Microsoft Edge";v="98"
sec-ch-ua-arch: "x86"
sec-ch-ua-full-version: "98.0.1108.50"
sec-ch-ua-mobile: ?0
sec-ch-ua-model: ""
sec-ch-ua-platform: "Windows"
sec-ch-ua-platform-version: "14.0.0"
sec-fetch-dest: document
sec-fetch-mode: navigate
sec-fetch-site: same-origin
sec-fetch-user: ?1
service-worker-navigation-preload: true
upgrade-insecure-requests: 1
user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36 Edg/98.0.1108.50""")
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

def send_html(url,headers,params):
    while 1:
        try:
            response = requests.get(url,
                                    headers=headers,params=params,timeout=(4,5))
            time.sleep(.1)
            return response.text
        except Exception as e:
            print(f'some error:{e}')
            time.sleep(1)
  # 随机字符串

def toint(t):
    if str(t).isdigit():
        return int(t)
    else:
        return 0


def baidu_translate(content):
    salt = str(random.randint(0, 50))
    # 申请网站 http://api.fanyi.baidu.com/api/trans
    appid = '20210926000957280'  # 这里写你自己申请的
    secretKey = 'jrb6WQieqfinLykPb9OV'  # 这里写你自己申请的
    sign = appid + content + salt + secretKey
    sign = hashlib.md5(sign.encode(encoding='UTF-8')).hexdigest()
    head = {'q': f'{content}',
            'from': 'auto',
            'to': 'zh',
            'appid': f'{appid}',
            'salt': f'{salt}',
            'sign': f'{sign}'}
    j = requests.get('http://api.fanyi.baidu.com/api/trans/vip/translate', head)
    res = j.json()
    trans_result = "\n".join([i.get("dst","") for  i in res.get("trans_result",[])])
    trans_form = res.get("from")
    print(f">>> 翻译后：{trans_result}")
    return trans_result,trans_form


def get_detail(vid,vau):
    # 检索方式为：
    print(f"vid:{vid}")
    api_uri = 'https://www.googleapis.com/youtube/v3/'
    video_info_search_url = f'{api_uri}videos?id={vid}&part=contentDetails,id,liveStreamingDetails,localizations,player,' \
                            f'recordingDetails,snippet,statistics,status,topicDetails&key={api_key}'
    res =  send_get(video_info_search_url)
    response = res.get("items")[0]
    cid = response.get("snippet").get("channelId")

    saveitem = {}
    saveitem["视频id"] = vid
    saveitem["作者"] = vau[1:]
    saveitem["发布时间"] = response.get("snippet",{}).get("publishedAt")
    saveitem["标题"] = response.get("snippet",{}).get("title")
    saveitem["视频简介"] = response.get("snippet",{}).get("description")
    saveitem["观看数"] = response.get("statistics",{}).get("viewCount")
    saveitem["评论数"] = toint(response.get("statistics",{}).get("commentCount"))
    saveitem["点赞数"] = response.get("statistics",{}).get("likeCount")
    saveitem["标题语言"] = response.get("snippet",{}).get("defaultAudioLanguage")
    saveitem["视频时长"] = response.get("contentDetails",{}).get("duration")
    saveitem["视频标签"] = ";".join(response.get("snippet",{}).get("tags",[]))
    saveitem["频道名称"] = response.get("snippet",{}).get("channelTitle")


    print(vid,saveitem)
    print(f"*" * 80)
    return saveitem





if  __name__ == "__main__":
    results = []
    records = pandas.read_excel("用户视频列表.xlsx",dtype=str).to_dict(orient='records')
    for vinfo in records:
        vid = vinfo.get("视频id")
        vau = vinfo.get("关键字")
        vdetail = get_detail(vid=vid,vau=vau)
        results.append(vdetail)
    df = pandas.DataFrame(sorted(results,key=lambda x:int(x.get("观看数")),reverse=True))
    df.to_excel("post_detail.xlsx",index=False)