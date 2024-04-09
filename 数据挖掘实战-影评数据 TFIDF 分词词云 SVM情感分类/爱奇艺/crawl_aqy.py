import re
import time

import copyheaders
import pandas
import requests
from lxml import etree

headers = copyheaders.headers_raw_to_dict(b"""
Accept:application/json, text/plain, */*
Accept-Language:zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6
Origin:https://www.iqiyi.com
Referer:https://www.iqiyi.com/ranks1PCW/-1/-6?vfrm=rank_list&vfrmblk=channel.-1.bk&vfrmrst=rank.more
Sec-Ch-Ua:"Microsoft Edge";v="123", "Not:A-Brand";v="8", "Chromium";v="123"
Sec-Ch-Ua-Mobile:?0
Sec-Ch-Ua-Platform:"Windows"
Sec-Fetch-Dest:empty
Sec-Fetch-Mode:cors
Sec-Fetch-Site:same-site
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0
X-B3-Traceid:a99e3798807bf3e04932be153b9fe053
""")


def send_get(url,headers,params):
    while 1:
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=(4,5)
            )
            time.sleep(2)
            return response
        except Exception as e:
            print(f'some error:{e}')
            time.sleep(1)

def start_crawl():
    for page in range(1,100):
        url = "https://mesh.if.iqiyi.com/portal/pcw/rankList/comSecRankList"
        params = {
            "v": "1",
            "device": "854325f24118b52ab795ba79af4cf7ac",
            "auth": "",
            "uid": "",
            "ip": "202.108.14.240",
            "refresh": "0",
            "server": "false",
            "page_st": "-6",
            "tag": "-6",
            "category_id": "-1",
            "date": "",
            "pg_num": page,
            "next_pos": ""
        }
        response = send_get(url,headers,params).json()

        contents = response.get("data",{}).get("items",[{}])[0].get("contents",[])

        print(f'page:{page} page length:{len(contents)}')


        for icontent in contents:
            """
            最高热度
            弹幕量
            影片（电视剧/综艺）名字
            影片板块：（电影/电视剧/综艺/动漫等）
            年代
            影片类型：（警匪、古装、喜剧等）
            演员：
            集数
            """
            saveitem = {}
            saveitem["影视id"] = icontent.get("tvid")
            saveitem["最高热度"] = icontent.get("mainIndex")
            saveitem["弹幕量"] = icontent.get("bulletIndex")
            saveitem["影片名字"] = icontent.get("title")
            saveitem["影片板块"] = icontent.get("tags").split("/")[0]
            saveitem["年代"] = icontent.get("tags").split("/")[1]
            saveitem["影片类型"] = icontent.get("tags").split("/")[2]
            saveitem["演员"] = icontent.get("tags").split("/")[3]

            saveitem["影片描述"] = icontent.get("desc")
            saveitem["页面网址"] = icontent.get("pageUrl")
            saveitem["封面图"] = icontent.get("imageCover")


            detail_response = send_get(saveitem["页面网址"],headers,{})

            js = ";".join(re.findall(r'"update_status":"(.*?)",',detail_response.text))
            saveitem["集数"] = js

            print(page,saveitem)
            movieinfo.append(saveitem)
            comment_url = "https://sns-comment.iqiyi.com/v3/comment/get_baseline_comments.action"
            comment_params = {
                "agent_type": "118",
                "business_type": "17",
                "content_id": icontent.get("tvid"),
                "need_vote": "1",
                "page": "2",
                "page_size": "20",
                "qyid": "5032eb745659236cfe7e529f94b88129",
                "sort": "HOT",
                "tail_num": "1"
            }

            for comment_page in range(1,6):

                comment_response = send_get(comment_url,headers,comment_params).json()

                comments = comment_response.get("data",{}).get("comments")

                comment_ids = []

                for icomment in comments:

                    comment_obj = {}
                    comment_obj["影视id"] = icontent.get("tvid")
                    comment_obj["影视名称"] = icontent.get("title")
                    comment_obj["评论id"] = icomment.get("id")
                    comment_obj["评论内"] = icomment.get("content")
                    comment_obj["评论时间"] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(icomment.get("addTime")))
                    comment_obj["评论回复数"] = icomment.get("replyCount")
                    comment_obj["评论点赞数"] = icomment.get("likes")
                    comment_obj["用户ip属地"] = icomment.get("location")
                    comment_obj["用户昵称"] = icomment.get("userInfo",{}).get("uname")
                    comment_obj["用户性别"] = icomment.get("userInfo",{}).get("gender")
                    comment_ids.append(icomment.get("id"))
                    print(comment_obj)
                    all_comments.append(comment_obj)
                if len(comments) < 10:
                    break
                comment_params['last_id'] = comment_ids[-1]
                print(f'下一页id：',comment_params['last_id'])

        if len(contents) < 10:
            break


if  __name__ == "__main__":
    all_comments = []
    movieinfo = []
    start_crawl()

    pandas.DataFrame(all_comments).to_excel("评论表.xlsx",index=False)
    pandas.DataFrame(movieinfo).to_excel("影视表.xlsx",index=False)
