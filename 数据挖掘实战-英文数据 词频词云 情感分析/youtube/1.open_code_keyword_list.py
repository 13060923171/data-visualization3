import json
import os
import re
import sys
import time
import copyheaders
import pandas
import requests
import langid
import re

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

headers = copyheaders.headers_raw_to_dict(b"""
accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
accept-language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6
cache-control: no-cache
cookie: VISITOR_INFO1_LIVE=pzNaW-5TpDs; PREF=tz=Asia.Shanghai; YSC=eI3HWxy5CYk; GPS=1
pragma: no-cache
sec-ch-ua: " Not A;Brand";v="99", "Chromium";v="98", "Microsoft Edge";v="98"
sec-ch-ua-arch: "x86"
sec-ch-ua-full-version: "98.0.1108.50"
sec-ch-ua-mobile: ?0
sec-ch-ua-model: ""
sec-ch-ua-platform: "Windows"
sec-ch-ua-platform-version: "14.0.0"
sec-fetch-dest: document
sec-fetch-mode: navigate
sec-fetch-site: same-origin
sec-fetch-user: ?1
service-worker-navigation-preload: true
upgrade-insecure-requests: 1
user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36 Edg/98.0.1108.50""")

jheaders = copyheaders.headers_raw_to_dict(b"""
accept: */*
accept-language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6
cache-control: no-cache
content-length: 2616
content-type: application/json
cookie: VISITOR_INFO1_LIVE=pzNaW-5TpDs; PREF=tz=Asia.Shanghai; YSC=eI3HWxy5CYk; GPS=1
origin: https://www.youtube.com
pragma: no-cache
referer: https://www.youtube.com/results?search_query=deepfake
sec-ch-ua: " Not A;Brand";v="99", "Chromium";v="98", "Microsoft Edge";v="98"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "Windows"
sec-fetch-dest: empty
sec-fetch-mode: same-origin
sec-fetch-site: same-origin
user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36 Edg/98.0.1108.50
x-goog-visitor-id: Cgtwek5hVy01VHBEcyi16qOQBg%3D%3D
x-youtube-client-name: 1
x-youtube-client-version: 2.20220211.01.00
""")


def check_language(string: str) -> str:
    """检查语言
    :return zh:中文,en:英文,
    """


    new_string = re.sub(r'[0-9]+', '', string)  # 这一步剔除掉文本中包含的数字
    return langid.classify(new_string)[0]

def handle_list_vinfo(listdata,keyword,c_c,maxvinfo,initdata_config):
    for item in listdata:
        try:

            if item.get("videoRenderer") == None:
                continue
            vid = item.get("videoRenderer").get("videoId")
            title = ' '.join([i.get("text") for i in item.get("videoRenderer").get("title").get("runs")])
            ##视频标签 视频简介 视频发布时间  标签 播放量 视频时长 点赞量
            time_str = item.get("videoRenderer").get("lengthText").get("simpleText")
            if len(time_str.split(":")) == 2:
                timetot = int(time_str.split(":")[0]) * 60 + int(time_str.split(":")[1])
            else:
                timetot = '-1'
            baseitem = {}
            baseitem["视频id"] = vid
            baseitem["关键字"] = keyword
            baseitem["描述"] = title
            baseitem["发布者名称"] = item.get("videoRenderer").get("ownerText").get("runs")[0].get("text")
            baseitem["发布者id"] = item.get("videoRenderer").get("ownerText").get("runs")[0]. \
                get("navigationEndpoint").get("commandMetadata").get("webCommandMetadata").get("url")
            baseitem["发布时间"] = item.get("videoRenderer").get("publishedTimeText").get("simpleText")
            baseitem["观看数"] = float(item.get("videoRenderer").get("viewCountText").get("simpleText").replace("次观看",'').replace(",",'').replace("无人观看",'0'))
            baseitem["时长"] = timetot
            baseitem["网址"] = f'https://www.youtube.com/watch?v={baseitem["视频id"]}'
            c_c.append(baseitem)

        except Exception as e:
            print(f">>>error: {e}")





def  getvinfo_by_key(keyword,maxvinfo):
    base_api = f'https://www.youtube.com/results?search_query={keyword}'
    res = requests.get(base_api,headers=headers,verify=False)
    initdata = re.findall(r'var ytInitialData = (.*?);</script>',res.text)
    if len(initdata) == 0:
        print(f"初始化信息为空 ==》")
        sys.exit()
    jsondata = json.loads(initdata[0])
    base_config = jsondata.get("contents").get("twoColumnSearchResultsRenderer").\
          get("primaryContents").get("sectionListRenderer").get("contents")[-1].get("continuationItemRenderer").get("continuationEndpoint")
    initdata_config = {"c":0}
    initdata_config["params"] = base_config.get("clickTrackingParams")
    initdata_config["token"] = base_config.get("continuationCommand").get("token")
    handle_list = jsondata.get("contents").get("twoColumnSearchResultsRenderer").\
          get("primaryContents").get("sectionListRenderer").get("contents")[0].get("itemSectionRenderer").get("contents")
    handle_list_vinfo(handle_list,keyword,c_c,maxvinfo,initdata_config)
    print(f'下一页游标：{initdata_config}')
    while True:
        base_api = 'https://www.youtube.com/youtubei/v1/search?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8'

        data = '{"context":{"client":{"hl":"zh-CN","gl":"HK","remoteHost":"2407:cdc0:a5fe:6f8:d7a9:3d5:b7ac:d0a1","deviceMake":"","deviceModel":"","visitorData":"Cgs4MU5mVVdMUlRGTSjVpKGpBjIICgJISxICGgA%3D","userAgent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36,gzip(gfe)","clientName":"WEB","clientVersion":"2.20231012.01.00","osName":"Windows","osVersion":"10.0","originalUrl":"https://www.youtube.com/results?search_query=&sp=CAMSBAgFEAE%253D","platform":"DESKTOP","clientFormFactor":"UNKNOWN_FORM_FACTOR","configInfo":{"appInstallData":"CNWkoakGEJfn_hIQvPmvBRC4-68FENShrwUQiOOvBRDqw68FEPOorwUQw_evBRCn6v4SENnurwUQ3ej-EhDZya8FEKT4rwUQp_evBRDi1K4FEJ_jrwUQqfevBRD1-a8FENPhrwUQtaavBRDM364FEL_3rwUQq4KwBRDM_68FELzrrwUQyPevBRCJ6K4FEK76rwUQ6-j-EhC0ya8FEMyu_hIQ57qvBRC--a8FENXlrwUQrLevBRDbr68FEOSz_hIQvbauBRC4i64FEKXC_hIQ-r6vBRDrk64FENfprwUQ6ej-EhC36v4SEJrwrwUQ7qKvBRCm7P4SEMH3rwUQn_T-EhDT9q8FENiAsAU%3D"},"userInterfaceTheme":"USER_INTERFACE_THEME_LIGHT","browserName":"Chrome","browserVersion":"118.0.0.0","acceptHeader":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7","deviceExperimentId":"ChxOekk0T1RFMk5qVXlNamd4T0RnNU1qY3pOUT09ENWkoakGGNWkoakG","screenWidthPoints":927,"screenHeightPoints":963,"screenPixelDensity":1,"screenDensityFloat":1,"utcOffsetMinutes":480,"connectionType":"CONN_CELLULAR_4G","memoryTotalKbytes":"8000000","mainAppWebInfo":{"graftUrl":"https://www.youtube.com/results?search_query=ti%E1%BA%BFng+Trung","pwaInstallabilityStatus":"PWA_INSTALLABILITY_STATUS_CAN_BE_INSTALLED","webDisplayMode":"WEB_DISPLAY_MODE_BROWSER","isWebNativeShareAvailable":true},"timeZone":"Asia/Shanghai"},"user":{"lockedSafetyMode":false},"request":{"useSsl":true,"internalExperimentFlags":[{"key":"force_enter_once_in_webview","value":"true"}],"consistencyTokenJars":[]},' \
               '"clickTracking":{"clickTrackingParams":"'+str(initdata_config["params"])+'"},"adSignalsInfo":{"params":[{"key":"dt","value":"1697141332866"},{"key":"flash","value":"0"},{"key":"frm","value":"0"},{"key":"u_tz","value":"480"},{"key":"u_his","value":"7"},{"key":"u_h","value":"1080"},{"key":"u_w","value":"1920"},{"key":"u_ah","value":"1040"},{"key":"u_aw","value":"1920"},{"key":"u_cd","value":"24"},{"key":"bc","value":"31"},{"key":"bih","value":"963"},{"key":"biw","value":"910"},{"key":"brdim","value":"0,0,0,0,1920,0,1920,1040,927,963"},{"key":"vis","value":"1"},{"key":"wgl","value":"true"},{"key":"ca_type","value":"image"}]}}' \
               ',"continuation":"'+str(initdata_config["token"])+'"}'
        while 1:
            try:
                resdetail = requests.post(base_api, headers=jheaders, data=data, verify=False).json()
                break
            except Exception as e:
                print(f"error:{e}")
                time.sleep(1)
        # time.sleep(1)
        base_items = resdetail.get("onResponseReceivedCommands")[0].get("appendContinuationItemsAction").get("continuationItems")
        result_list = base_items[0].get("itemSectionRenderer").get("contents")
        try:
            handle_list_vinfo(result_list,keyword,c_c,maxvinfo,initdata_config)
        except :
            continue
        try:
            initdata_config["params"] = base_items[-1].get("continuationItemRenderer").get("continuationEndpoint").get(
                "clickTrackingParams")
        except Exception as e:
            print(f'暂无下一页：exit:{e}')
            break
        if initdata_config["c"] >= maxvinfo:
            print(f">>>超出maxinfo限制exit")
            break
        initdata_config["token"] = base_items[-1].get("continuationItemRenderer").get("continuationEndpoint").get("continuationCommand").get("token")
        print(f'下一页游标：{initdata_config}')

    pd_ = pandas.DataFrame(sorted(c_c,key=lambda x:x.get("观看数"),reverse=True))
    pd_.drop_duplicates(['视频id'],inplace=True)
    pd_.to_excel(f"post_list.xlsx",index=False)




if  __name__ == "__main__":
    ##走vpn节点
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    c_c = []
    getvinfo_by_key(keyword='florasis',maxvinfo=5000)

