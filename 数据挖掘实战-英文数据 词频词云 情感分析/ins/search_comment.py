import json
import os
import random
import time

import copyheaders
import pandas
import requests

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
class Com(object):
    start_count = 0

    def     __init__(self):
        df = pandas.read_excel("post_data.xlsx")
        self.records = df.to_dict(orient='records')[self.start_count:]
        self.headers   = copyheaders.headers_raw_to_dict(b"""
Accept:*/*
Cookie:mid=ZYBpCwALAAHWZgow0GsjWfUUn_8R; ig_did=43FA2380-A8D9-4DF8-B876-C87EBA00C62C; ig_nrcb=1; datr=AWmAZRc7aXJFh5E59rktbjaL; ps_l=0; ps_n=0; csrftoken=O8AbIFqcG26GaF6ZpH1MOXlM2EGpKXQr; ds_user_id=65111820987; sessionid=65111820987%3AUNrrBTVZFbpRZV%3A22%3AAYe-KxYAElMLygfOo7SbYgKIzy6te5dfjlQdTZjWzA; rur="EAG\05465111820987\0541741026963:01f77f24d0b0ce672a792cc9e8185bc3af03e3ce7b4234842d94265a0c7cac2686d26824"
Dpr:1
Pragma:no-cache
Referer:https://www.instagram.com/p/C0yU8x_OKp0/
Sec-Ch-Prefers-Color-Scheme:light
Sec-Ch-Ua:"Chromium";v="122", "Not(A:Brand";v="24", "Microsoft Edge";v="122"
Sec-Ch-Ua-Full-Version-List:"Chromium";v="122.0.6261.70", "Not(A:Brand";v="24.0.0.0", "Microsoft Edge";v="122.0.2365.59"
Sec-Ch-Ua-Mobile:?0
Sec-Ch-Ua-Model:""
Sec-Ch-Ua-Platform:"Windows"
Sec-Ch-Ua-Platform-Version:"10.0.0"
Sec-Fetch-Dest:empty
Sec-Fetch-Mode:cors
Sec-Fetch-Site:same-origin
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0
Viewport-Width:947
X-Asbd-Id:129477
X-Csrftoken:O8AbIFqcG26GaF6ZpH1MOXlM2EGpKXQr
X-Ig-App-Id:936619743392459
X-Ig-Www-Claim:hmac.AR3r_aBH0OOOH-paTUByGVuKJuWaVO9kRso1bxOA84n1GAsy
X-Requested-With:XMLHttpRequest
""")


    def _download(self,item):
        api = f'https://i.instagram.com/api/v1/media/{item["帖子id"]}/comments/'
        pageConfig = {
            "after": '',
        }

        for page in range(1):
            params = {
                "can_support_threading":True,
                "min_id": '{"is_server_cursor_inverse":true,"server_cursor":"' + pageConfig["after"] + '"}'
            }
            while 1:
                try:
                    res = requests.get(
                        api,
                        headers=self.headers,
                        params=params,
                        timeout=(4,5)
                    ).json()
                    time.sleep(random.uniform(1,2))
                    break
                except Exception as e:
                    print(f">>> 网络： {e}")
                    time.sleep(1)
            print(res)
            for comment in res["comments"]:
                try:
                    saveitem = item.copy()
                    saveitem["评论ID"] = comment.get("pk")
                    saveitem["评论时间"] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(comment["created_at"]))
                    saveitem["评论文本"] = comment.get("text")
                    saveitem["评论喜欢数"] = comment.get("comment_like_count")
                    print(self.start_count,page,saveitem["评论文本"],saveitem)
                    with open("temp.txt",'a',encoding='utf-8') as f:
                        f.write(json.dumps(saveitem))
                        f.write('\n')
                except Exception as e:
                    print(f">>>error: cms {e}")
                    time.sleep(1)
            try:
                next_min_id = json.loads(res['next_min_id'])["server_cursor"]
                pageConfig["after"] = next_min_id
                flag = json.loads(res['next_min_id'])["is_server_cursor_inverse"]
            except Exception as e:
                print(f">>> 解析下一页失败：{e}")
                return
            if len(res["comments"])<= 9 or not flag:
                break

    def parse(self):
        for rec in self.records:
            self.start_count += 1

            print(f">>> 正在访问：{rec['帖子id'] }  {self.start_count} ")
            if rec.get("评论数") <=0 :
                continue
            try:
                self._download(rec)
            except Exception as e:
                print(f">>> error: {e}")
                time.sleep(10)




if __name__ == "__main__":
    Com().parse()
    with open("temp.txt",'r',encoding='utf-8') as f:
        lines = [json.loads(i.strip()) for  i in f.readlines()]
        df = pandas.DataFrame(lines)
        df.to_excel("comment.xlsx",index=False)
    os.remove("temp.txt")