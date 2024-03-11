import json
import os
import random
import re
import time

import copyheaders
import langid
import pandas
import requests

headers = copyheaders.headers_raw_to_dict(b"""
Accept:*/*
Content-Type:application/x-www-form-urlencoded
Cookie:mid=ZYBpCwALAAHWZgow0GsjWfUUn_8R; ig_did=43FA2380-A8D9-4DF8-B876-C87EBA00C62C; ig_nrcb=1; datr=AWmAZRc7aXJFh5E59rktbjaL; ps_l=0; ps_n=0; csrftoken=O8AbIFqcG26GaF6ZpH1MOXlM2EGpKXQr; ds_user_id=65111820987; sessionid=65111820987%3AUNrrBTVZFbpRZV%3A22%3AAYe-KxYAElMLygfOo7SbYgKIzy6te5dfjlQdTZjWzA; rur="EAG\05465111820987\0541741025717:01f7dd41a45413342be707f47d01304e220b45a0ddca72b0a15c9c862a18ae550f1b26b5"
Dpr:1
Origin:https://www.instagram.com
Pragma:no-cache
Referer:https://www.instagram.com/sanxingduiculture/
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
X-Fb-Friendly-Name:PolarisProfilePostsTabContentQuery_connection
X-Fb-Lsd:zs2Nsjnxj-WI_Z_YH-EhgS
X-Ig-App-Id:936619743392459
""")

v_result = []


def check_language(string: str) -> str:
    """检查语言
    :return zh:中文,en:英文,
    """
    if string == "":
        return 'en'

    new_string = re.sub(r'[0-9]+', '', str(string))  # 这一步剔除掉文本中包含的数字
    return langid.classify(new_string)[0]


def mian(username, userid):
    api = f'https://www.instagram.com/api/graphql'
    cursor = {"cursor": 'null'}
    for page in range(1, 100):
        body = {
            "av": "17841465189618977",
            "__d":'www',
            "__user":'0',
            "__a": "1",
            "__req": "o",
            "__hs": "19785.HYP:instagram_web_pkg.2.1..0.1",
            "dpr": "1",
            "__ccg": "UNKNOWN",
            "__rev": "1011800519",
            "__s": "8v5atc:w8bvyg:l6uh5m",
            "__hsi": "7342202406259166227",
            "__dyn": "7xeUjG1mxu1syUbFp40NonwgU7SbzEdF8aUco2qwJxS0k24o0B-q1ew65xO0FE2awlU-cw5Mx62G3i1ywOwv89k2C1Fwc60D8vw8OfK0EUjwGzEaE7622362W2K0zK5o4q3y1Sx-220gO2Sq2-azo7u3u2C2O0z8c86-3u2WE5B0bK1Iwqo5q1IQp1yUoxeubxKi2K7E",
            "__csr": "ggMl6996gViYjqi9bq9uBtqfunJiaGyGjqGDGGVblKn-i8F6AWjCLUOcAoCbgFfgBalpVbHyW8jBVbVpqV4AvHhpF-m5QnqGegS9Gqiirm5aDyqCoGcBFxe8Uy5Fo01kScM2EggwAw7xwNgy0dMw1BCqowCkwSki0guHwXwwwC4kmfwr404ZcU16k0g102Jo1scM2yw09ci",
            "__comet_req": "7",
            "fb_dtsg": "NAcN0ghS-c0zvzx7GyELkSWtEpzoRjgZUFEGihwBv0vggjwi8vdVrnA:17853599968089360:1709489687",
            "jazoest": "26637",
            "lsd": "zs2Nsjnxj-WI_Z_YH-EhgS",
            "__spin_r": "1011800519",
            "__spin_b": "trunk",
            "__spin_t": "1709489712",
            "fb_api_caller_class": "RelayModern",
            "fb_api_req_friendly_name": "PolarisProfilePostsTabContentQuery_connection",
            "variables": r'{"after":"xxxx1","before":null,"data":{"count":12,"include_relationship_info":true,"latest_besties_reel_media":true,"latest_reel_media":true},"first":12,"last":null,"username":"xxxx2","__relay_internal__pv__PolarisShareMenurelayprovider":false}',
            "server_timestamps": "true",
            "doc_id": "7784658434954494"
        }
        if page == 1:
            body["variables"] = body["variables"].replace('"xxxx1"', 'null').replace("xxxx2",username)
        else:
            body["variables"] = body["variables"].replace("xxxx1", cursor["cursor"]).replace("xxxx2", username)

        res = requests.post(
            api,
            headers=headers,
            data=body,
        ).json()
        time.sleep(random.uniform(3, 4))
        results = res.get("data", {}).get("xdt_api__v1__feed__user_timeline_graphql_connection", {}).get("edges", [])
        if results is None:
            results = []
        print(page, len(results))
        for ire in results:
            iml = ire.get("node", {})
            saveinfo = {}
            saveinfo["用户"] = username
            saveinfo["其他"] = iml.get("accessibility_caption")
            saveinfo['点赞数'] = iml.get("like_count")
            saveinfo['评论数'] = iml.get('comment_count')
            saveinfo['帖子id'] = iml.get("id")
            saveinfo["帖子正文"] = iml.get("caption", {}).get("text") if iml.get("caption", {}) != None else ''
            saveinfo["创建时间"] = time.strftime('%Y-%m-%d', time.localtime(iml.get("taken_at")))

            v_result.append(saveinfo)
            print(page, saveinfo["帖子正文"][:20], saveinfo)

        if len(results) < 4:
            break
        cursor["cursor"] = res.get("data", {}).get("xdt_api__v1__feed__user_timeline_graphql_connection", {}).get("page_info", {}).get("end_cursor")
        print(cursor["cursor"] )
        if cursor["cursor"]  is None:
            break



if __name__ == "__main__":
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    #mallariangelique----zs782#5wMP!g----henduohao.com/2fa/6Z4U6N75UXZJ7VNNQIOLNL3AHBXSWY3M

    mian(username='sanxingduiculture', userid='49159372088')
    mian(username='center.sichuan', userid='50668186245')

    pandas.DataFrame(v_result).to_excel("post_data.xlsx", index=False)
