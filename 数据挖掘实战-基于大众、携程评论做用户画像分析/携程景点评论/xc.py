import json
import os
import time

import copyheaders
import pandas
import requests
from lxml import etree

headers = copyheaders.headers_raw_to_dict(b"""
accept: */*
cookie: ibulanguage=CN; ibulocale=zh_cn; cookiePricesDisplayed=CNY; Union=OUID=&AllianceID=5376&SID=167068&SourceID=&createtime=1679399698&Expires=1680004497700; MKT_OrderClick=ASID=5376167068&AID=5376&CSID=167068&OUID=&CT=1679399697701&CURL=https%3A%2F%2Fhotels.ctrip.com%2F%3Fsid%3D167068%26allianceid%3D5376%26qh_keywordid%3D3483236709%26qh_creative%3D3453155524%26qh_planid%3D1875088274%26qh_unitid%3D1069672110%26qh_device%3Dpc%26qhclickid%3D2b7b2ed942767f42%26keywordid%3D3483236709&VAL={}; GUID=09031137417573003236; MKT_CKID=1679399697993.95neg.42ih; MKT_CKID_LMT=1679399697994; _RF1=36.157.97.126; _RSG=CYa_27Nzyo5nucEDnBsNY9; _RDG=2833fc40a4363b22a908eb380b54dda016; _RGUID=5cbcf571-c2fe-488c-9897-d8bc4d3a6e31; _bfaStatusPVSend=1; nfes_isSupportWebP=1; Hm_lvt_e4211314613fcf074540918eb10eeecb=1679399702; nfes_isSupportWebP=1; ASP.NET_SessionSvc=MTAuMTEzLjkyLjk0fDkwOTB8b3V5YW5nfGRlZmF1bHR8MTYzODQzNDIwNzc1NQ; MKT_Pagesource=H5; librauuid=; _pd=%7B%22r%22%3A1%2C%22_d%22%3A230%2C%22_p%22%3A6%2C%22_o%22%3A53%2C%22s%22%3A290%2C%22_s%22%3A0%7D; Hm_lpvt_e4211314613fcf074540918eb10eeecb=1679401070; _bfa=1.1679399697634.2zs282.1.1679399697634.1679399951189.1.35.10650064020; _bfs=1.32; _ubtstatus=%7B%22vid%22%3A%221679399697634.2zs282%22%2C%22sid%22%3A1%2C%22pvid%22%3A35%2C%22pid%22%3A290546%7D; _jzqco=%7C%7C%7C%7C1679399697953%7C1.2087740640.1679399697728.1679401046048.1679401071516.1679401046048.1679401071516.undefined.0.0.31.31; __zpspc=9.1.1679399697.1679401071.31%233%7Ccn.bing.com%7C%7C%7C%7C%23; _bfi=p1%3D290546%26p2%3D290546%26v1%3D35%26v2%3D34; _bfaStatus=success
user-agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Mobile Safari/537.36
""")


def send_req(api):
    while 1:
        try:
            res = requests.get(
                api, headers=headers, timeout=(4, 5)
            )
            time.sleep(0.5)
            return res.text

        except Exception as e:
            print(f" error :{e} ")
            time.sleep(1)


def download_cms(sightid, sightname):
    headers = copyheaders.headers_raw_to_dict(b"""
    accept: */*
    content-type: application/json
    cookie: _RGUID=668f7630-ce37-4a26-aa1c-6e4edf206c78; _RSG=ceqDE3ZTFQ6un68VxaRSeA; _RDG=2809cfc3e53af220033a608ef9c6c7a25f; ibulanguage=CN; ibulocale=zh_cn; cookiePricesDisplayed=CNY; MKT_CKID=1689674330096.n7qit.q2yd; _bfaStatusPVSend=1; nfes_isSupportWebP=1; nfes_isSupportWebP=1; UBT_VID=1689674329320.xp5j7; _ga=GA1.2.1889109162.1693035524; FlightIntl=Search=[%22SHA|%E4%B8%8A%E6%B5%B7(SHA)|2|SHA|480%22%2C%22TYO|%E4%B8%9C%E4%BA%AC(TYO)|228|TYO|540%22%2C%222023-09-03%22%2C%222023-09-06%22]; GUID=52271140496414499452; __zpspc=9.25.1694625557.1694628170.36%233%7Ccn.bing.com%7C%7C%7C%7C%23; _lizard_LZ=HbmGPld3DRznJCaL6fcWQFy9j0YoUhBIXwANxkgiMTtsuOvSZVKE+p174-q528re; login_type=0; login_uid=33E433AAD183A2BA50A29B2CE481E672; _RF1=36.157.105.119; _ga_5DVRDQD429=GS1.2.1697534525.3.1.1697534558.0.0.0; _ga_B77BES1Z8Z=GS1.2.1697534525.3.1.1697534558.27.0.0; intl_ht1=h4=2_425587,1_61840052,2_75220482,1_108715215,1_110934165,1_110518137; Union=OUID=title&AllianceID=3958554&SID=23830328&SourceID=&createtime=1698122977&Expires=1698727777334; MKT_OrderClick=ASID=395855423830328&AID=3958554&CSID=23830328&OUID=title&CT=1698122977335&CURL=https%3A%2F%2Fwww.ctrip.com%2F%3Fsid%3D23830328%26allianceid%3D3958554%26ouid%3Dtitle&VAL={}; MKT_CKID_LMT=1698122980589; MKT_Pagesource=PC; _bfa=1.1689674329320.xp5j7.1.1698123146854.1698123194883.48.5.0; _ubtstatus=%7B%22vid%22%3A%221689674329320.xp5j7%22%2C%22sid%22%3A48%2C%22pvid%22%3A5%2C%22pid%22%3A0%7D; _bfi=p1%3D290510%26p2%3D290510%26v1%3D5%26v2%3D2; _bfaStatus=success; _jzqco=%7C%7C%7C%7C1698122981190%7C1.1665914166.1693034741989.1698123140865.1698123196032.1698123140865.1698123196032.undefined.0.0.165.165
    cookieorigin: https://m.ctrip.com
    origin: https://m.ctrip.com
    referer: https://m.ctrip.com/webapp/you/commentWeb/commentList?seo=0&businessId=1936675&businessType=11&hideStatusBar=1&openapp=5&poiId=28408248&noJumpApp=yes&from=https%3A%2F%2Fgs.ctrip.com%2Fhtml5%2Fyou%2Fsight%2Fshanghai2%2F1936675.html%3Fseo%3D1
    user-agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Mobile Safari/537.36
    """)
    comment_list = []
    api = f'https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList?_fxpcqlniredt=52271140496414499452&x-traceID=52271140496414499452-1698124799969-4824086'
    for itag in [0,-11,-12,121110475936,221110475936,321110475936,421110475936,521110475936,621110475936]:
        for page in range(1, 301):
            try:

                data = {"arg": {"channelType": 2, "collapseType": 0, "commentTagId": itag, "pageIndex": page, "pageSize": 10,
                                "poiId": sightid, "sourceType": 1, "sortType": 3, "starType": 0},
                        "head": {"cid": "52271140496414499452", "ctok": "", "cver": "1.0", "lang": "01", "sid": "8888",
                                "syscode": "09", "auth": "", "xsid": "", "extension": []}}
                res = requests.post(api, headers=headers, data=json.dumps(data)).json()
                totpage = res.get("result").get("totalCount") // 10 +1
                result = res['result']["items"]
                if result is None:
                    result = []
                for dataff in result:
                    saveitem = {}
                    saveitem["sight_name"] = sightname
                    saveitem["sightid"] = sightid
                    saveitem["commentId"] = dataff.get("commentId")
                    saveitem["userNick"] = dataff.get("userInfo", {}).get("userNick")
                    saveitem["userMember"] = dataff.get("userInfo", {}).get("userMember")
                    saveitem["replyInfo"] = dataff.get("userInfo", {}).get("replyInfo")
                    saveitem["commentKeywordList"] = dataff.get("commentKeywordList")
                    saveitem["commentTagInfo"] = dataff.get("commentTagInfo")
                    saveitem["publishStatus"] = dataff.get("publishStatus")
                    saveitem["usefulCount"] = dataff.get("usefulCount")
                    saveitem["replyCount"] = dataff.get("replyCount")
                    saveitem["score"] = dataff.get("score")
                    saveitem["images"] = ";".join([i.get("imageSrcUrl") for i in dataff.get("images")])
                    saveitem["scores"] = ";".join(f'{i.get("name")}-{i.get("score")}' for i in dataff.get("scores"))
                    saveitem["content"] = dataff.get("content")
                    saveitem["recommendItems"] = dataff.get("recommendItems")
                    saveitem["publishTypeTag"] = dataff.get("publishTypeTag")
                    saveitem["childrenTag"] = dataff.get("childrenTag")
                    saveitem["ipLocatedName"] = dataff.get("ipLocatedName")
                    print(sightname, page, saveitem)
                    comment_list.append(saveitem)
                if len(result) < 10 or page >= totpage:
                    break
                time.sleep(.5)
                print(res)
            except Exception as e:
                print(f" some error:{e}")
                time.sleep(1)

    pandas.DataFrame(comment_list).to_excel(f'携程-{sightname}.xlsx',index=False)

if __name__ == "__main__":
    # download_cms(sightid="75936",sightname='布达拉宫')
    download_cms(sightid="75938",sightname='大昭寺')
#
