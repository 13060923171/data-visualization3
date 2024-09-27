import json
import os.path
import re
import time
import urllib.parse
from hashlib import md5
import copyheaders
import pandas
import requests
import random

session = requests.Session()  # 创建一个session对象


headers = copyheaders.headers_raw_to_dict(b"""
Accept:*/*
Accept-Language:zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6
Cookie:buvid3=1E43DBF9-0CEC-A056-18A2-ADB00572733746997infoc; b_nut=1721887846; _uuid=410B9B6DA-7210F-B547-E94C-D6A155DBB1DD47638infoc; CURRENT_FNVAL=4048; enable_web_push=DISABLE; header_theme_version=CLOSE; rpdid=|(k|k)k~uYkk0J'u~ku)mu|JY; DedeUserID=68451030; DedeUserID__ckMd5=e48931738cbc5eac; buvid4=91C0FB8B-1BB6-F29C-A652-332BF6DADDE886244-024031810-oB32s6Vja4U%2FMNrIf5opJA%3D%3D; buvid_fp_plain=undefined; fingerprint=24ce0d030a2719787a28ce2d0cdc202e; hit-dyn-v2=1; home_feed_column=5; browser_resolution=1912-954; bp_t_offset_68451030=978457016760532992; buvid_fp=24ce0d030a2719787a28ce2d0cdc202e; SESSDATA=25c745e3%2C1742564261%2Cec024%2A91CjBfp2woapRfToRYcwLTWZmMPpMLwj6U3KJpWNaUUbAym1BwsP51rd8F6sx9zjfWsFASVmVQbWhmOXVnSmpiZzlXcndZYmlhVElvcDkxdnNUT3gyTHJ1WGt2NWdhanVJcG5FcXRHd09wVlNxUkhXQjlXNENMV0UwLTd5eGk0OWRhb2dLam1KdllRIIEC; bili_jct=65b6c39e0ecbc83477c7b5888ebf465e; sid=57go7m73; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MjcyNzE0NjMsImlhdCI6MTcyNzAxMjIwMywicGx0IjotMX0.lTPCon3CKkmDg4VSlDH_0-M10xSiT3Cr9XTiS0QVpt4; bili_ticket_expires=1727271403; b_lsid=3BE4FF18_19224088875; share_source_origin=COPY; bsource=share_source_copy_link
Origin:https://www.bilibili.com
Priority:u=1, i
Referer:https://www.bilibili.com/opus/921595772937437201?spm_id_from=333.999.0.0
Sec-Ch-Ua:"Chromium";v="124", "Microsoft Edge";v="124", "Not-A.Brand";v="99"
Sec-Ch-Ua-Mobile:?0
Sec-Ch-Ua-Platform:"Windows"
Sec-Fetch-Dest:empty
Sec-Fetch-Mode:cors
Sec-Fetch-Site:same-site
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0
""")


def validateTitle(title):
    re_str = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(re_str, "_", title)  # 替换为下划线
    return new_title


def send_get(url, headers, params):
    while 1:
        try:
            print(f"访问：{url}")
            response = session.get(
                url,
                headers=headers,
                params=params,
                timeout=(4, 5)
            )
            print(response.text)
            time.sleep(random.uniform(1,3))
            if '请求被拦截' in response.text:
                print('请求被拦截')
                time.sleep(10)
                continue
            return response.json()
        except Exception as e:
            print(f"some error:{e}")
            time.sleep(1)


def md5_use(text: str) -> str:
    result = md5(bytes(text, encoding="utf-8")).hexdigest()
    return result

def main(aid,vid):

    print(f'aid:{aid}')

    page_cursor = {"p":'{"offset":""}'}

    for page in range(1, 300):

        url = "https://api.bilibili.com/x/v2/reply/wbi/main"
        params = {
            "oid": aid,
            "type": 1,
            "mode": "3",
            "pagination_str": page_cursor["p"],
            "plat": "1",
            "seek_rpid": "0",
            "web_location": "1315875",
            "wts": '1716005768'
        }

        sub_enc = f'mode=3&oid={params["oid"]}&pagination_str={urllib.parse.quote(page_cursor["p"])}&plat=1&seek_rpid=0&type={params["type"]}&web_location=1315875&wts={params["wts"]}'
        w_rid = md5_use(sub_enc+ 'ea1db124af3c7062474693fa704f4ff8')
        params['w_rid'] = w_rid

        response = send_get(url, headers, params=params)
        try:

            replies = response.get("data", {}).get("replies", [])
            if replies is None:
                replies = []
            print(aid, page, len(replies))
        except Exception as e:
            print(f'some error:{e}')
            print(response)
            time.sleep(10)
            break
        for reply in replies:
            try:
                saveitem = {}
                saveitem["视频vid"] = vid
                saveitem["视频aid"] = aid

                saveitem["评论内容"] = reply.get("content", {}).get("message")
                saveitem["创建时间"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reply.get("ctime")))
                saveitem["评论点赞数"] = reply.get("like")
                saveitem["评论用户id"] = reply.get("member").get("mid")
                saveitem["评论用户昵称"] = reply.get("member").get("uname")
                saveitem["评论ip属地"] = reply.get("reply_control").get("location")

                saveitem["评论用户性别"] = reply.get("member").get("sex")
                saveitem["评论用户等级"] = reply.get("member").get("level_info", {}).get('current_level')
                saveitem["评论用户是否为认证用户"] = reply.get("member").get("official_verify").get("desc")
                saveitem["评论用户签名"] = reply.get("member").get("sign")
                saveitem['图片列表'] = "; ".join([i.get("img_src") for i in reply.get("content",{}).get("pictures",[])])
                saveitem["子评论数"] = reply.get("rcount")
                saveitem["评论楼层"] = '一级评论'
                print(aid, page, saveitem)
                all_comments.append(saveitem)

                if reply.get("rcount") <= 3:


                    for detail_rev in reply.get('replies', []):
                        saveitem = {}
                        saveitem["视频vid"] = vid

                        saveitem["视频aid"] = aid
                        saveitem["评论内容"] = detail_rev.get("content", {}).get("message")
                        saveitem["创建时间"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(detail_rev.get("ctime")))
                        saveitem["评论点赞数"] = detail_rev.get("like")
                        saveitem["评论用户id"] = detail_rev.get("member").get("mid")
                        saveitem["评论用户昵称"] = detail_rev.get("member").get("uname")
                        saveitem["评论ip属地"] = detail_rev.get("reply_control").get("location")
                        saveitem["评论用户性别"] = detail_rev.get("member").get("sex")
                        saveitem["评论用户等级"] = detail_rev.get("member").get("level_info", {}).get('current_level')
                        saveitem["评论用户是否为认证用户"] = detail_rev.get("member").get("official_verify").get("desc")
                        saveitem["评论用户签名"] = detail_rev.get("member").get("sign")
                        saveitem['图片列表'] = "; ".join([i.get("img_src") for i in detail_rev.get("content", {}).get("pictures", [])])
                        saveitem["子评论数"] = detail_rev.get("rcount")
                        saveitem["评论楼层"] = '二级评论'

                        print(f'下一等级评论：', aid, page, saveitem)

                        all_comments.append(saveitem)

                else:

                    for sub_page in range(1,reply.get("rcount")//10 + 2):
                        sub_url = "https://api.bilibili.com/x/v2/reply/reply"
                        sub_params = {
                            "oid": aid,
                            "type": 1,
                            "root": reply.get("rpid_str"),
                            "ps": "10",
                            "pn": sub_page,
                            "gaia_source": "main_web",
                            "web_location": "333.1369",
                            "w_rid": "c570ea40a106402882dba26d46e25a58",
                            "wts": "1716003408"
                        }

                        sub_response = send_get(sub_url, headers, params=sub_params)
                        try:

                            sub_replies = sub_response.get("data", {}).get("replies", [])
                            if sub_replies is None:
                                sub_replies = []
                            print(aid, sub_page, len(sub_replies))
                        except Exception as e:
                            print(f'some error:{e}')
                            print(sub_response)
                            time.sleep(10)
                            break
                        for detail_rev in sub_replies:
                            saveitem = {}
                            saveitem["视频vid"] = vid

                            saveitem["视频aid"] = aid
                            saveitem["评论内容"] = detail_rev.get("content", {}).get("message")
                            saveitem["创建时间"] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(detail_rev.get("ctime")))
                            saveitem["评论点赞数"] = detail_rev.get("like")
                            saveitem["评论用户id"] = detail_rev.get("member").get("mid")
                            saveitem["评论用户昵称"] = detail_rev.get("member").get("uname")
                            saveitem["评论ip属地"] = detail_rev.get("reply_control").get("location")

                            saveitem["评论用户性别"] = detail_rev.get("member").get("sex")
                            saveitem["评论用户等级"] = detail_rev.get("member").get("level_info", {}).get('current_level')
                            saveitem["评论用户是否为认证用户"] = detail_rev.get("member").get("official_verify").get("desc")
                            saveitem["评论用户签名"] = detail_rev.get("member").get("sign")
                            saveitem['图片列表'] = "; ".join([i.get("img_src") for i in detail_rev.get("content", {}).get("pictures", [])])
                            saveitem["子评论数"] = detail_rev.get("rcount")
                            saveitem["评论楼层"] = '二级评论'

                            print(f'下一等级评论：', aid, sub_page, saveitem)

                            all_comments.append(saveitem)

                        if len(sub_replies) < 2:
                            break
            except Exception as e:
                print(f"parse error comment:{e}")
        if len(replies) < 2:
            break
        else:
            cursor_ = response.get("data",{}).get("cursor",{}).get("pagination_reply",{}).get("next_offset")
            cursor = '{"offset":'+str(json.dumps(cursor_))+'}'
            print(f">>> next page :{cursor}")
            page_cursor["p"] = cursor
        if response.get("data",{}).get("cursor",{}).get("is_end"):
            break








if __name__ == '__main__':

    all_comments = []


    allvideo = pandas.read_csv("./config/video.csv",dtype=str).to_dict(orient='records')
    for iv in allvideo:

        main(aid=iv.get("aid"),vid=iv.get("bvid"))




    df = pandas.DataFrame(all_comments)
    df.drop_duplicates(inplace=True)
    df.to_excel(f"评论表.xlsx", index=False)