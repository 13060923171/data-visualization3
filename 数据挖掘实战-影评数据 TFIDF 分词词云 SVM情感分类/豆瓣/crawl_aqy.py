import re
import time

import copyheaders
import pandas
import requests
from lxml import etree

headers = copyheaders.headers_raw_to_dict(b"""
Accept:application/json, text/plain, */*
Cookie:bid=G-ycMAGCqRY; ll="118269"; Hm_lvt_6d4a8cfea88fa457c3127e14fb5fabc2=1711353067,1711356660,1711432517; _ga=GA1.1.1842630230.1711353067; _ga_Y4GN1R87RG=GS1.1.1711434386.5.0.1711434386.0.0.0; __utma=30149280.682352180.1711170861.1712047146.1712163922.12; __utmc=30149280; __utmz=30149280.1712163922.12.3.utmcsr=cn.bing.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmt=1; __utmb=30149280.1.10.1712163922; ap_v=0,6.0; __gads=ID=905a37d7312491c4:T=1711619496:RT=1712163925:S=ALNI_MYvMM_H6DuEcBa8m0KVKZoEu2z0gA; __gpi=UID=00000d716da837c9:T=1711619496:RT=1712163925:S=ALNI_MZrMdFhPfwrq6f7kpgJJbLVJgbx0w; __eoi=ID=375c1f9e43f61b69:T=1711619496:RT=1712163925:S=AA-AfjZCyzA5c0-16o43avFg562w
Host:m.douban.com
Origin:https://movie.douban.com
Referer:https://movie.douban.com/tv/
Sec-Ch-Ua:"Microsoft Edge";v="123", "Not:A-Brand";v="8", "Chromium";v="123"
Sec-Ch-Ua-Mobile:?0
Sec-Ch-Ua-Platform:"Windows"
Sec-Fetch-Dest:empty
Sec-Fetch-Mode:cors
Sec-Fetch-Site:same-site
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0
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
            time.sleep(.2)
            return response
        except Exception as e:
            print(f'some error:{e}')
            time.sleep(1)

def start_crawl():

    for tvtp in ['2020年代','2024','2023','2022','2021','2020','2019','2010年代','2000年代']:
        for page in range(0, 6):
            url = "https://m.douban.com/rexxar/api/v2/tv/recommend"
            params = {
                "refresh": "0",
                "start": page * 20,
                "count": "20",
                "selected_categories": "{}",
                "uncollect": "false",
                "tags": tvtp
            }
            response = send_get(url, headers, params).json()

            contents = response.get("items", [])

            print(f'{tvtp},page:{page} page length:{len(contents)}')

            for icontent in contents:
                try:
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

                    if icontent.get("rating") is None:
                        icontent['rating'] = {}

                    saveitem = {}
                    saveitem["影视id"] = icontent.get("id")
                    saveitem["影片名字"] = icontent.get("title")

                    saveitem["封面图"] = icontent.get("pic", {}).get("large")
                    saveitem["总评论数"] = icontent.get("rating", {}).get("count")
                    saveitem["评论分数"] = icontent.get("rating", {}).get("value")
                    saveitem["详情链接"] = f"https://m.douban.com/movie/subject/{icontent.get('id')}/"

                    detail_response = send_get(
                        f'https://m.douban.com/rexxar/api/v2/tv/{icontent.get("id")}?ck=&for_mobile=1', headers,
                        {}).json()

                    if detail_response.get("directors") is None:
                        detail_response['directors'] = []

                    if detail_response.get("actors") is None:
                        detail_response['actors'] = []

                    saveitem["发布年份"] = detail_response.get("pubdate")
                    saveitem["发布地区"] = detail_response.get("countries")
                    saveitem["类型"] = detail_response.get("genres")
                    saveitem["导演"] = [i.get("name") for i in detail_response.get("directors")]
                    saveitem["主演"] = [i.get("name") for i in detail_response.get("actors")]
                    saveitem["集数"] = detail_response.get("episodes_count")
                    saveitem["影片简介"] = detail_response.get("intro")
                    saveitem["影片语言"] = detail_response.get("languages")
                    saveitem["影评长评数"] = detail_response.get("review_count")

                    print(tvtp, page, saveitem)
                    movieinfo.append(saveitem)

                    for comment_page in range(0, 3):
                        comment_url = f"https://m.douban.com/rexxar/api/v2/tv/{icontent.get('id')}/interests"
                        comment_params = {
                            "count": "20",
                            "order_by": "hot",
                            "anony": "0",
                            "start": comment_page * 20,
                            "ck": "",
                            "for_mobile": "1"
                        }
                        comment_response = send_get(comment_url, headers, comment_params).json()

                        comments = comment_response.get("interests")

                        for icomment in comments:
                            try:
                                comment_obj = {}
                                comment_obj["影视id"] = icontent.get('id')
                                comment_obj["影视名称"] = icontent.get("title")
                                comment_obj["评论id"] = icomment.get("id")
                                comment_obj["评论内容"] = icomment.get("comment")
                                comment_obj["评论评分"] = icomment.get("rating", {}).get("value")
                                comment_obj["评论时间"] = icomment.get("create_time")
                                comment_obj["评论点赞数"] = icomment.get("vote_count")
                                comment_obj["用户ip属地"] = icomment.get("ip_location")
                                comment_obj["用户昵称"] = icomment.get("user", {}).get("name")
                                comment_obj["用户性别"] = icomment.get("user", {}).get("gender")
                                print(tvtp, comment_page, comment_obj)
                                all_comments.append(comment_obj)
                            except Exception as e:
                                print(f'parse error:{e}')
                        if len(comments) < 10:
                            break
                except Exception as e:
                    print(f'parse item movie error:{e}')

            if len(contents) < 10:
                break


if  __name__ == "__main__":
    all_comments = []
    movieinfo = []
    start_crawl()

    pandas.DataFrame(all_comments).to_excel("评论表.xlsx",index=False)
    pandas.DataFrame(movieinfo).to_excel("影视表.xlsx",index=False)
