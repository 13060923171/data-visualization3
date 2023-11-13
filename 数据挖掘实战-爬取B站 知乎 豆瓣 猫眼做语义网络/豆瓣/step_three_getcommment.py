import json
import os
import pprint
import time

import copyheaders
import pandas
import redis
import requests

headers = copyheaders.headers_raw_to_dict(b"""
Accept:application/json
Cookie:bid=9D4A2T4Pk5I; ll="118269"; __utmz=30149280.1694613332.7.6.utmcsr=cn.bing.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __gads=ID=9fbf2f98980772bc-22f782e54f800028:T=1690207878:RT=1694613337:S=ALNI_MYtgSUkpFQf0gdpz5acr0xQG6h8vw; __gpi=UID=00000d125d7f5aa5:T=1690207879:RT=1694613337:S=ALNI_MaId_kDXR5YmV7enoCQDhsDUTotZg; ap_v=0,6.0; __utma=30149280.11078882.1690207643.1696680292.1698906492.9; __utmb=30149280.0.10.1698906492; __utmc=30149280; Hm_lvt_6d4a8cfea88fa457c3127e14fb5fabc2=1696680422,1696683537,1698906519; _ga=GA1.2.728625589.1696680423; _gid=GA1.2.1807619520.1698906519; _ga_Y4GN1R87RG=GS1.1.1698906519.3.0.1698906524.0.0.0; Hm_lpvt_6d4a8cfea88fa457c3127e14fb5fabc2=1698906525
Host:m.douban.com
Pragma:no-cache
Referer:https://m.douban.com/movie/subject/1292052/comments?sort=new_score&start=25
Sec-Ch-Ua:"Not/A)Brand";v="99", "Microsoft Edge";v="115", "Chromium";v="115"
Sec-Ch-Ua-Mobile:?1
Sec-Ch-Ua-Platform:"Android"
Sec-Fetch-Dest:empty
Sec-Fetch-Mode:cors
Sec-Fetch-Site:same-origin
User-Agent:Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Mobile Safari/537.36 Edg/115.0.1901.203
X-Requested-With:XMLHttpRequest
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
            time.sleep(.6)
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

    for page in range(0,400):
        offset = page *  25
        url = f'https://m.douban.com/rexxar/api/v2/movie/{vid}/interests?count=20&order_by=hot&anony=0&start={offset}&ck=&for_mobile=1'
        response = send_get(url)
        comments = response.get("interests",[])

        for item in comments:
            try:
                saveitem = record.copy()
                saveitem["评论内容"] = item.get("comment")
                saveitem["评论id"] = item.get("id")
                saveitem["评论性别"] = item.get("user", {}).get("gender")
                saveitem["ip属地"] = item.get("ip_location")
                saveitem["评论者"] = item.get("user", {}).get("name")
                saveitem["打分"] = item.get("rating", {}).get("value")
                saveitem["时间"] = item.get("create_time")
                saveitem["有用数"] = item.get("vote_count")
                print(config_number["p"], page, saveitem)
                with open("temp_comment.txt", 'a', encoding='utf-8') as f:
                    f.write(json.dumps(saveitem))
                    f.write('\n')
            except Exception as e:
                print(f"parse error:{e}")

        if len(comments) < 20:
            break


# except Exception as e:
#     print(f"parse error:{e}")
#     time.sleep(1)


if __name__ == "__main__":
    # records = pandas.read_excel("电影表.xlsx", dtype=str).to_dict(orient='records')
    # config_number = {"p":0}
    # for record in records:
    #     config_number["p"]+=1
    #     get_detail(record=record)
    #
    with open("temp_comment.txt",'r',encoding='utf-8') as f:
        comments = [json.loads(i.strip()) for i in f.readlines()]
    df = pandas.DataFrame(comments)
    df.to_excel("评论下载.xlsx",index=False)