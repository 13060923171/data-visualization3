import json
import os
import pprint
import time

import copyheaders
import pandas
import redis
import requests

headers = copyheaders.headers_raw_to_dict(b"""
Accept:application/json, text/plain, */*
Content-Type:application/json
Cookie:sajssdk_2015_cross_new_user=1; searchHistoryCookie=%u5510%u4EBA%u8857%u63A2%u68483%2C%u590D%u4EC7%u8005%u8054%u76DF4%u7EC8%u5C40%u4E4B%u6218%2C%u957F%u6D25%u6E56%u4E4B%u6C34%u95E8%u6865%2C%u6D41%u6D6A%u5730%u74032%2C%u7EA2%u6D77%u884C%u52A8; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2218b8ecd4051a87-022485716d366e4-7c54647e-2073600-18b8ecd4052d40%22%2C%22first_id%22%3A%22%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMThiOGVjZDQwNTFhODctMDIyNDg1NzE2ZDM2NmU0LTdjNTQ2NDdlLTIwNzM2MDAtMThiOGVjZDQwNTJkNDAifQ%3D%3D%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%22%2C%22value%22%3A%22%22%7D%2C%22%24device_id%22%3A%2218b8ecd4051a87-022485716d366e4-7c54647e-2073600-18b8ecd4052d40%22%7D; Hm_lvt_07aa95427da600fc217b1133c1e84e5b=1698907899,1698909081; Hm_lpvt_07aa95427da600fc217b1133c1e84e5b=1698909097
Host:front-gateway.mtime.com
Origin:http://movie.mtime.com
Pragma:no-cache
Referer:http://movie.mtime.com/
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.203
X-Mtime-Wap-Checkvalue:mtime
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

    for page in range(1,400):
        url = f'http://front-gateway.mtime.com/library/movie/comment.api?tt={int(time.time())*1000}&movieId={vid}&pageIndex={page}&pageSize=20&orderType=1'
        response = send_get(url)
        comments = response.get("data",{}).get("list",[])

        for item in comments:
            try:
                saveitem = record.copy()
                saveitem["评论内容"] = item.get("content")
                saveitem["评论id"] = item.get("commentId")
                saveitem["评论性别"] = "-"
                saveitem["ip属地"] = item.get("location")
                saveitem["评论者"] = item.get("nickname")
                saveitem["打分"] =item.get("rating")
                saveitem["时间"] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime( item.get("commentTime")))
                saveitem["有用数"] = item.get("praiseCount")
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
    df["评论内容"] = df["评论内容"].apply(lambda x: str(x).encode('UTF-8', 'ignore').decode('UTF-8'))
    df.to_excel("评论下载.xlsx",index=False)