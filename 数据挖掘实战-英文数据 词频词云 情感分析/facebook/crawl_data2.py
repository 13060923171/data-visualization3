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
Cookie:sb=GOl7ZQn4tigYyKWA7r9lxgkA; datr=GOl7ZT_Pid-KjEVP6a0KktPM; c_user=100077428293552; ps_n=0; dpr=2; locale=en_US; vpd=v1%3B810x324x2; wl_cbv=v2%3Bclient_version%3A2419%3Btimestamp%3A1709123350; xs=27%3A48zKJWQXYsuSAQ%3A2%3A1705302393%3A-1%3A-1%3A%3AAcWxMBrnfeTbZMxgUtUGMCS3V8zF2qMLpj33HbdU2g; fr=1TDOWz13ViSoWl4AI.AWVMyueG2ae11A-IBsgk7rwqAxw.Bl5LbS..AAA.0.0.Bl5LbS.AWW97W5ik6Q; wd=943x919; presence=C%7B%22t3%22%3A%5B%5D%2C%22utc3%22%3A1709487913463%2C%22v%22%3A1%7D
Dpr:1
Origin:https://www.facebook.com
Referer:https://www.facebook.com/profile.php?id=100077961559653
Sec-Ch-Prefers-Color-Scheme:light
Sec-Ch-Ua:"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"
Sec-Ch-Ua-Full-Version-List:"Chromium";v="122.0.6261.95", "Not(A:Brand";v="24.0.0.0", "Google Chrome";v="122.0.6261.95"
Sec-Ch-Ua-Mobile:?0
Sec-Ch-Ua-Model:""
Sec-Ch-Ua-Platform:"Windows"
Sec-Ch-Ua-Platform-Version:"10.0.0"
Sec-Fetch-Dest:empty
Sec-Fetch-Mode:cors
Sec-Fetch-Site:same-origin
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36
Viewport-Width:943
X-Asbd-Id:129477
X-Fb-Friendly-Name:ProfileCometTimelineFeedRefetchQuery
X-Fb-Lsd:h4X_Qnw3YXmh3K52SSRNfp
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



    def runkeyword(self,username,userid,idate):
        start_time = int(time.mktime(time.strptime(idate,'%Y-%m-%d')))
        end_time = int(time.mktime(time.strptime(f'{idate} 23:59:59','%Y-%m-%d %H:%M:%S')))

        cursor = {"cursor":'null'}
        for page in range(1,2000):

            api = 'https://www.facebook.com/api/graphql/'

            body = {
                "av": "100077428293552",
                "__user": "100077428293552",
                "__a": "1",
                "__req": "v",
                "__hs": "19785.HYP:comet_pkg.2.1..2.1",
                "dpr": "1",
                "__ccg": "EXCELLENT",
                "__rev": "1011800216",
                "__s": "3zc2vb:uhi5kj:iuhk30",
                "__hsi": "7342194666552626171",
                "__dyn": "7AzHK4HwkEng5K8G6EjBAg2owIxu13wFG14xt3odE98K361twYwJyE24wJwpUe8hwaG1sw9u0LVEtwMw65xO2OU7m221FwgolzUO0-E4a3a4oaEnxO0Bo7O2l2Utwwwi831wiE567Udo5qfK0zEkxe2GewyDwkUtxGm2SUbElxm3y11xfxmu3W3y261eBx_wHwdG7FoarCwLyES1Iwh888cA0z8c84q58jyUaUcojxK2B08-269wkopg6C13whEeE4WVU-4Edouw",
                "__csr": "gbAr4PEnkp2AbRsO9OmAxcr5hG44Gy9lQYyB9SDSqOmyZTvEOiiy5hvGGjOaQhe8FtafRABV49BhE-pkKrLGykaQWKFolrAGm9zoRfGmmFEy8ADzmdxaiGVuVEKby8LzFWAnHyUR1Cegao8UrxeqiAeCG5FohCo9EmyEylxaA6oSi5U9VUeo8o4q2K7okzV8C7oWE4q6UizUy1sALUtw_wRwww8-2m2Siawu8fU8EjwRwOwq82Kws87a0oS6o9o1N6AU1UE8k2C0rm0YXwdW092w0pNU03Apxe0CE0SK0JUdQ3G0EA09Hw2d80qSU0Si0i20jSaw_w22U9U09hkdAQ3iq02_G01otg4qppe0FE7y0NU",
                "__comet_req": "15",
                "fb_dtsg": "NAcM6fMVbqe3H-xpIQLoHzvlZWQKUIw6SHzfjIUOyqkwi6USutXIomA:27:1705302393",
                "jazoest": "25697",
                "lsd": "h4X_Qnw3YXmh3K52SSRNfp",
                "__aaid": "0",
                "__spin_r": "1011800216",
                "__spin_b": "trunk",
                "__spin_t": "1709487909",
                "fb_api_caller_class": "RelayModern",
                "fb_api_req_friendly_name": "ProfileCometTimelineFeedRefetchQuery",
                "variables": r'{"UFI2CommentsProvider_commentsKey":"ProfileCometTimelineRoute","afterTime":xxxx4,"beforeTime":xxxx3,"count":3,"cursor":"xxxx1","displayCommentsContextEnableComment":null,"displayCommentsContextIsAdPreview":null,"displayCommentsContextIsAggregatedShare":null,"displayCommentsContextIsStorySet":null,"displayCommentsFeedbackContext":null,"feedLocation":"TIMELINE","feedbackSource":0,"focusCommentID":null,"memorializedSplitTimeFilter":null,"omitPinnedPost":true,"postedBy":{"group":"OWNER"},"privacy":null,"privacySelectorRenderLocation":"COMET_STREAM","renderLocation":"timeline","scale":1,"stream_count":1,"taggedInOnly":null,"useDefaultActor":false,"id":"xxxx2","__relay_internal__pv__IsWorkUserrelayprovider":false,"__relay_internal__pv__IsMergQAPollsrelayprovider":false,"__relay_internal__pv__CometUFIReactionsEnableShortNamerelayprovider":false,"__relay_internal__pv__CometUFIIsRTAEnabledrelayprovider":false,"__relay_internal__pv__StoriesArmadilloReplyEnabledrelayprovider":false,"__relay_internal__pv__StoriesRingrelayprovider":true}',
                "server_timestamps": "true",
                "doc_id": "7456926777661832"
            }
            if  page == 1:
                body["variables"] = body["variables"].replace('"xxxx1"','null').replace("xxxx2",userid).replace('xxxx3',str(end_time)).replace('xxxx4',str(start_time))
            else:
                body["variables"] = body["variables"].replace("xxxx1", cursor["cursor"]).replace("xxxx2", userid).replace('xxxx3',str(end_time)).replace('xxxx4',str(start_time))


            response = self.send_post(api,self.headers,body).split("\n")
            print(f'[{idte}]',username,page,len(response))
            for res in response:
                if res.startswith('{"data":'):
                    results = json.loads(res).get("data",{}).get("node",{}).get("timeline_list_feed_units")

                    edges = results.get("edges",[])
                    print(f'[{idte}]',username,page,'edges',len(edges))

                    if len(edges)    == 0:
                        return
                    for iedge in edges:
                        try:
                            story  = iedge.get("node").get("comet_sections").get(
                                "content").get("story")
                            feedback_context = iedge.get("node").get("comet_sections").get(
                                "feedback").get("story").get("feedback_context")
                            feedback = feedback_context.get("feedback_target_with_context").get("ufi_renderer").get("feedback").get("comet_ufi_summary_and_actions_renderer").get("feedback")
                            net_url = iedge.get("node").get("comet_sections").get("feedback").get("story").get("shareable_from_perspective_of_feed_ufi").get("url")
                            saveitem = {}


                            saveitem["帖子id"] = feedback.get("id")

                            try:
                                saveitem["正文"] = story.get("message").get("text")
                            except:
                                saveitem["正文"] = ""
                            saveitem["时间"] = ""
                            try:
                                timesamps = iedge.get("node").get("comet_sections").get("context_layout").get(
                                    "story").get("comet_sections").get("metadata")
                                for itams  in timesamps:
                                    if itams.get('story').get('creation_time') is not None:
                                        saveitem["时间"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(itams.get('story').get('creation_time')))
                                        break
                            except:
                                pass
                            saveitem["帖子链接"] = net_url
                            saveitem["点赞数"] = feedback.get("reaction_count").get("count")
                            saveitem["评论数"] = feedback.get("total_comment_count")
                            saveitem["分享数"] = feedback.get("share_count").get("count")



                            print(f'[{idte}]',username,page,saveitem["时间"],saveitem)

                            with open(f"{username}_data.txt", 'a', encoding='utf-8') as f:
                                f.write(json.dumps(saveitem))
                                f.write('\n')
                        except Exception as e:
                            print(f"some error:{e}")
                            time.sleep(1)

                    if edges[-1].get("cursor") is not None:

                        cursor["cursor"] = edges[-1].get("cursor")
                        print(f'[{idte}]',f"下一页：",cursor["cursor"])
                    else:
                        print(f'[{idte}]',f">>> 暂无下一页")
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

    # 推文内容	含图片/视频/转载内容（是/否）	点赞数	评论数	转发数	发布时间 链接

    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"



    tl = CrawlFaceBook()
    #
    # for idte in getdaterange('2000-01-01','2024-02-29'):
    #     tl.runkeyword('SanxingduiCulture','100077961559653',idte)

    # for idte in getdaterange('2000-01-01','2024-02-29'):
    #     tl.runkeyword('center.sichuan','100076190284448',idte)


    newdata = []
    for user in ['center.sichuan','SanxingduiCulture']:
        file = f'./{user}_data.txt'
        with open(file,'r',encoding='utf-8') as f:
            lines = [json.loads(i.strip()) for i in f.readlines()]
            for ilin in lines:
                ilin["user"] = user
                if user != 'center.sichuan':
                    newdata.append(ilin)
                    continue
                if user== 'center.sichuan' and 'sanxingdui' in str(ilin["正文"]).lower():
                    newdata.append(ilin)

    pandas.DataFrame(newdata).to_excel("finall.xlsx",index=False)
#


