import json
import os
import re
import time
import copyheaders
import langid
import pandas
import requests


class CrawlFaceBook():


    def __init__(self):

        self.headers = copyheaders.headers_raw_to_dict(b"""
Accept:*/*
Content-Type:application/x-www-form-urlencoded
Cookie:sb=GOl7ZQn4tigYyKWA7r9lxgkA; datr=GOl7ZT_Pid-KjEVP6a0KktPM; c_user=100077428293552; ps_n=0; vpd=v1%3B810x324x2; wl_cbv=v2%3Bclient_version%3A2419%3Btimestamp%3A1709123350; xs=27%3A48zKJWQXYsuSAQ%3A2%3A1705302393%3A-1%3A-1%3A%3AAcWSC1zl3WAi41HG8ZomRdS1jPuMYnprX6Rvgd4CsNU; fr=1NPjUlhe9ZRud5Dt5.AWXyuars5EdtoXvFih-pQuaICnY.Bl6qEV..AAA.0.0.Bl6qEi.AWUGTQBbPkw; wd=764x919; presence=C%7B%22t3%22%3A%5B%5D%2C%22utc3%22%3A1709876354282%2C%22v%22%3A1%7D
Dpr:1
Origin:https://www.facebook.com
Referer:https://www.facebook.com/profile.php?id=100076190284448
Sec-Ch-Prefers-Color-Scheme:light
Sec-Ch-Ua:"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"
Sec-Ch-Ua-Full-Version-List:"Chromium";v="122.0.6261.112", "Not(A:Brand";v="24.0.0.0", "Google Chrome";v="122.0.6261.112"
Sec-Ch-Ua-Mobile:?0
Sec-Ch-Ua-Model:""
Sec-Ch-Ua-Platform:"Windows"
Sec-Ch-Ua-Platform-Version:"10.0.0"
Sec-Fetch-Dest:empty
Sec-Fetch-Mode:cors
Sec-Fetch-Site:same-origin
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36
Viewport-Width:764
X-Asbd-Id:129477
X-Fb-Friendly-Name:CommentListComponentsRootQuery
X-Fb-Lsd:tB4UQ9mBWyY8vgwX9NEIp_
        """)




    def send_post(self,url,headers,data):
        while 1:
            try:
                response = requests.post(url,headers=headers,data=data,timeout=(4,5))
                time.sleep(1)
                if response.status_code != 200:
                    print(response.text)
                    time.sleep(10)
                    continue
                return response.text
            except Exception as e:
                print(f'send post error:{e}')
                time.sleep(2)



    def runkeyword(self,ipost):
        pid = ipost.get("帖子id")
        cursor = {"cursor":'null'}
        for page in range(1,40):

            api = 'https://www.facebook.com/api/graphql/'


            if  page == 1:
                body = {
                    "av": "100077428293552",
                    "__user": "100077428293552",
                    "__a": "1",
                    "__req": "2k",
                    "__hs": "19790.HYP2:comet_pkg.2.1..2.1",
                    "dpr": "1",
                    "__ccg": "EXCELLENT",
                    "__rev": "1011920236",
                    "__s": "pxoe9c:wtrz5g:chmg2d",
                    "__hsi": "7343863012067988585",
                    "__dyn": "7AzHK4HwkEng5K8G6EjBAg2owIxu13wFG14xt3odE98K361twYwJyE24wJwpUe8hwaG1sw9u0LVEtwMw65xO2OU7m221Fwgo9oO0-E4a3a4oaEnxO0Bo7O2l2Utwwwi831wiE567Udo5qfK0zEkxe2GewyDwkUtxGm2SUbElxm3y11xfxmu3W3y261eBx_wHwdG7FoarCwLyES1Iwh888cA0z8c84q58jyUaUcojxK2B08-269wkopg6C13whEeE4WVU-4Edouw",
                    "__csr": "gcschticr4MT5NBOq7iH8JiiOWkXZidlvRTT9O-Xlnibq9aAjQZnuypanFt5BiSZkteJADgG8mBHzrXF6FCJKGpe9JKcGGykAWjhK-4QehGCWGEC6eiamuA5bAiKmEpGmjyGyUy5A5oky-VbCgqUS7A5u2yqqrKdGq4ry8mixG2615Dz8y5u2ObxGm2GEqxqbwhF842eyUoxbwBwFx6262q8xamdwBx-11g426awqEy1lxO0CE2YGaw4tG0Mo0haw2eEa81VU0DG0c6w8G0H5w0Z6w046pg0KB1no0AOq11g2kw3OQ0oq04C2xq0CU6O0Uo5G039-043o3mw37k06PE0h9w0U0wjO0e-0Gr8-10w16e06c8",
                    "__comet_req": "15",
                    "fb_dtsg": "NAcORc67pinYpIvlNcY0nLLlRMMGg8_z2gZGV4qeir8uQIWz1CDzt-A:27:1705302393",
                    "jazoest": "25492",
                    "lsd": "tB4UQ9mBWyY8vgwX9NEIp_",
                    "__aaid": "0",
                    "__spin_r": "1011920236",
                    "__spin_b": "trunk",
                    "__spin_t": "1709876351",
                    "fb_api_caller_class": "RelayModern",
                    "fb_api_req_friendly_name": "CommentListComponentsRootQuery",
                    "server_timestamps": "true",
                    "doc_id": "7249620448487557"
                }
                rr = '{"commentsIntentToken":"RECENT_ACTIVITY_INTENT_V1","feedLocation":"TAHOE","feedbackSource":41,"focusCommentID":null,"scale":1,"useDefaultActor":false,"id":"xxxx2"}'
                body["variables"] =rr.replace("xxxx2",pid)
            else:
                body = {
                    "av": "100077428293552",
                    "__user": "100077428293552",
                    "__a": "1",
                    "__req": "2k",
                    "__hs": "19790.HYP2:comet_pkg.2.1..2.1",
                    "dpr": "1",
                    "__ccg": "EXCELLENT",
                    "__rev": "1011920236",
                    "__s": "pxoe9c:wtrz5g:chmg2d",
                    "__hsi": "7343863012067988585",
                    "__dyn": "7AzHK4HwkEng5K8G6EjBAg2owIxu13wFG14xt3odE98K361twYwJyE24wJwpUe8hwaG1sw9u0LVEtwMw65xO2OU7m221Fwgo9oO0-E4a3a4oaEnxO0Bo7O2l2Utwwwi831wiE567Udo5qfK0zEkxe2GewyDwkUtxGm2SUbElxm3y11xfxmu3W3y261eBx_wHwdG7FoarCwLyES1Iwh888cA0z8c84q58jyUaUcojxK2B08-269wkopg6C13whEeE4WVU-4Edouw",
                    "__csr": "gcschticr4MT5NBOq7iH8JiiOWkXZidlvRTT9O-Xlnibq9aAjQZnuypanFt5BiSZkteJADgG8mBHzrXF6FCJKGpe9JKcGGykAWjhK-4QehGCWGEC6eiamuA5bAiKmEpGmjyGyUy5A5oky-VbCgqUS7A5u2yqqrKdGq4ry8mixG2615Dz8y5u2ObxGm2GEqxqbwhF842eyUoxbwBwFx6262q8xamdwBx-11g426awqEy1lxO0CE2YGaw4tG0Mo0haw2eEa81VU0DG0c6w8G0H5w0Z6w046pg0KB1no0AOq11g2kw3OQ0oq04C2xq0CU6O0Uo5G039-043o3mw37k06PE0h9w0U0wjO0e-0Gr8-10w16e06c8",
                    "__comet_req": "15",
                    "fb_dtsg": "NAcORc67pinYpIvlNcY0nLLlRMMGg8_z2gZGV4qeir8uQIWz1CDzt-A:27:1705302393",
                    "jazoest": "25492",
                    "lsd": "tB4UQ9mBWyY8vgwX9NEIp_",
                    "__aaid": "0",
                    "__spin_r": "1011920236",
                    "__spin_b": "trunk",
                    "__spin_t": "1709876351",
                    "fb_api_caller_class": "RelayModern",
                    "fb_api_req_friendly_name": "CommentsListComponentsPaginationQuery",
                    "server_timestamps": "true",
                    "doc_id": "8178681625495054"
                }
                rr = '{"commentsAfterCount":-1,"commentsAfterCursor":"xxxx1","commentsBeforeCount":null,"commentsBeforeCursor":null,"commentsIntentToken":"RECENT_ACTIVITY_INTENT_V1","feedLocation":"DEDICATED_COMMENTING_SURFACE","focusCommentID":null,"scale":1,"useDefaultActor":false,"id":"xxxx2"}'
                body["variables"] = rr.replace("xxxx1", cursor["cursor"]).replace("xxxx2",pid)

            print(body["variables"])
            response = self.send_post(api,self.headers,body).split("\n")
            print(page,len(response))
            for res in response:
                if res.startswith('{"data":'):
                    results = json.loads(res).get("data",{}).get("node",{}).get("comment_rendering_instance_for_feed_location").get("comments")

                    edges = results.get("edges",[])
                    print(page,'edges',len(edges))

                    if len(edges)    == 0:
                        return
                    for iedge in edges:
                        feedback = iedge.get("node")
                        try:
                            if feedback.get("user") is None:
                                feedback['user']= {}
                            saveitem = ipost.copy()
                            saveitem["评论时间"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(feedback.get("created_time")))
                            saveitem["评论id"] = feedback.get("id")
                            saveitem["评论者id"] = feedback.get("user").get("id")
                            saveitem["评论者名称"] = feedback.get("user").get("name")
                            try:
                                saveitem["评论文本"] = feedback.get("body").get("text")
                            except:
                                saveitem["评论文本"] = ""

                            try:
                                saveitem["评论子评论数"] = feedback.get("feedback").get("total_comment_count")
                            except:
                                saveitem["评论子评论数"] = ""

                            try:
                                saveitem["评论点赞数"] = feedback.get("feedback").get("reactors").get("count")
                            except:
                                saveitem["评论点赞数"] = ""

                            print(saveitem["评论者id"],page,saveitem)

                            with open(f"comment_data.txt", 'a', encoding='utf-8') as f:
                                f.write(json.dumps(saveitem))
                                f.write('\n')
                        except Exception as e:
                            print(f"some error:{e}")
                            time.sleep(1)

                    if results.get("page_info",{}).get("has_next_page"):

                        cursor["cursor"] = results.get("page_info",{}).get("end_cursor")
                        print(f"下一页：",cursor["cursor"])
                    else:
                        print(f">>> 暂无下一页")
                        return


def check_language(string: str) -> str:
    """检查语言
    :return zh:中文,en:英文,
    """
    if string == "":
        return 'en'

    new_string = re.sub(r'[0-9]+', '', string)  # 这一步剔除掉文本中包含的数字
    return langid.classify(new_string)[0]


def getdaterange(starttime, endtime):
    time_str = []
    start_samp = time.mktime(time.strptime(starttime, '%Y-%m-%d'))
    while True:
        time_str.append(
            time.strftime('%Y-%m-%d', time.localtime(start_samp))
        )
        start_samp += 24 * 60 * 60
        if start_samp > time.mktime(time.strptime(endtime, '%Y-%m-%d')):
            break
    return time_str[::-1]

if  __name__ == "__main__":


    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"



    tl = CrawlFaceBook()

    all_post = pandas.read_excel("finall.xlsx").to_dict(orient='records')[:71]
    for ipost in all_post:
        if ipost.get("评论数") <= 0:
            continue
        try:
            tl.runkeyword(ipost)

        except Exception as e:
            print(f'some error:{e}')



    newdata = []
    with open(f'./comment_data.txt','r',encoding='utf-8') as f:
            lines = [json.loads(i.strip()) for i in f.readlines()]
            newdata+= lines
    pandas.DataFrame(newdata).to_excel("评论统计.xlsx",index=False)


