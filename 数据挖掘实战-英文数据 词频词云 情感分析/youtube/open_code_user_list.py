import json
import os
import re
import sys
import time
import copyheaders
import pandas
import requests


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

all_max_vinfo = 1000

c_c = []

page_config = {"index":0}
def handle_list_vinfo(listdata,keyword):
    for firinfo in listdata:
        try:
            item = firinfo.get("itemSectionRenderer").get("contents")[0].get("videoRenderer",{})
            page_config["index"] += 1
            vid = item.get("videoId")
            title = ' '.join([i.get("text") for i in item.get("title").get("runs")])
            ##视频标签 视频简介 视频发布时间  标签 播放量 视频时长 点赞量
            time_str = item.get("lengthText").get("simpleText")
            if len(time_str.split(":")) == 2:
                timetot = int(time_str.split(":")[0]) * 60 + int(time_str.split(":")[1])
            else:
                timetot = '-1'
            baseitem = {}
            baseitem["视频id"] = vid
            baseitem["关键字"] = keyword
            baseitem["描述"] = title
            baseitem["发布时间"] = item.get("publishedTimeText").get("simpleText")
            baseitem["观看数"] = float(item.get("viewCountText").get("simpleText").replace("次观看",'').replace(",",'').replace("无人观看",'0'))
            baseitem["时长"] = timetot
            baseitem["网址"] = f'https://www.youtube.com/watch?v={baseitem["视频id"]}'
            print(f"符合条件数：{len(c_c)} ：正在访问：{baseitem}")
            c_c.append(baseitem)
        except Exception as e:
            print(f">>>error: {e}")





def  getvinfo_by_key(keyword,maxvinfo):
    base_api = f'https://www.youtube.com/{keyword}/search?query=sanxingdui'
    res = requests.get(base_api,headers=headers,verify=False)
    initdata = re.findall(r'var ytInitialData = (.*?);</script>',res.text)
    if len(initdata) == 0:
        print(f"初始化信息为空 ==》")
        sys.exit()
    jsondata = json.loads(initdata[0])
    base_config = jsondata.get("contents").get("twoColumnBrowseResultsRenderer").get("tabs")[-1].get("expandableTabRenderer").get("content").get("sectionListRenderer").get("contents")\
    [-1].get("continuationItemRenderer",{}).get("continuationEndpoint",{})
    initdata_config = {}

    handle_list = jsondata.get("contents").get("twoColumnBrowseResultsRenderer").get("tabs")[-1].get("expandableTabRenderer").get("content").get("sectionListRenderer").get("contents")
    handle_list_vinfo(handle_list,keyword)

    try:
        initdata_config["params"] = base_config.get("clickTrackingParams")
        initdata_config["token"] = base_config.get("continuationCommand").get("token")
    except:
        return
    print(f'下一页游标：{initdata_config}')

    while True:
        base_api = 'https://www.youtube.com/youtubei/v1/browse?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8&prettyPrint=false'
        data = '{"context":{"client":{"hl":"zh-CN","gl":"TW","remoteHost":"2406:8dc0:5905:571f:d50a:b063:c2a2:bc4f"' \
               ',"deviceMake":"","deviceModel":"","visitorData":"Cgs4MU5mVVdMUlRGTSiXmJeqBjIICgJISxICGgA%3D","userAg' \
               'ent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.' \
               '0.0 Safari/537.36,gzip(gfe)","clientName":"WEB","clientVersion":"2.20231101.05.00","osName":"Windows","osV' \
               'ersion":"10.0","originalUrl":"https://www.youtube.com/watch?v=ZQrOIZgGbgQ","platform":"DESKTOP","clientForm' \
               'Factor":"UNKNOWN_FORM_FACTOR","configInfo":{"appInstallData":"CJeYl6oGEOuTrgUQ29ivBRDd6P4SEMyu_hIQzN-uB' \
               'RCa8K8FENyNsAUQqfevBRC9tq4FEL75rwUQ1YiwBRCyxq8FEKL4_hIQt--vBRCrgrAFENDirwUQ6sOvBRDr6P4SEN_YrwUQ4' \
               '9ivBRCD368FELfq_hIQtaavBRClwv4SENnJrwUQk_r-EhC_968FEJTZ_hIQ0-GvBRDV5a8FEKaBsAUQ1-mvBRC9i7' \
               'AFEPyFsAUQ3IKwBRD1-a8FENShrwUQvoqwBRD6vq8FEKf3rwUQieiuBRCogbAFELTJrwUQodevBRDJ968FELz' \
               'rrwUQiIewBRDbr68FELz5rwUQuIuuBRCI468FEOLUrgUQ57qvBRCu1P4SEPeJsAUQrLevBRDp6P4SEOSz_hIQ7qKvBRCs2a' \
               '8F"},"userInterfaceTheme":"USER_INTERFACE_THEME_LIGHT","timeZone":"Asia/Shanghai","browserName":"Chr' \
               'ome","browserVersion":"119.0.0.0","acceptHeader":"text/html,application/xhtml+xml,application/' \
               'xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7","d' \
               'eviceExperimentId":"ChxOekk1TnpRMk16RTNNalEwTXpRMU5UUXpOZz09EJeYl6oGGJeYl6oG","screenWidthPoints' \
               '":653,"screenHeightPoints":967,"screenPixelDensity":1,"screenDensityFloat":1,"utcOffsetMinutes":480,' \
               '"connectionType":"CONN_CELLULAR_4G","memoryTotalKbytes":"8000000","mainAppWebInfo":{"graftUrl":"https' \
               '://www.youtube.com/'+str(getvinfo_by_key)+'/videos","pwaInstallabilityStatus":"PWA_INSTALLABILITY_STATUS_CAN_BE_INST' \
               'ALLED","webDisplayMode":"WEB_DISPLAY_MODE_BROWSER","isWebNativeShareAvailable":true}},' \
               '"user":{"lockedSafetyMode":false},"request":{"useSsl":true,"internalExperimentFlags":[],"c' \
               'onsistencyTokenJars":[]},"clickTracking":{"clickTrackingParams":"'+str(initdata_config["params"])+'"},"adSignalsInfo":{"params":[{"key":"dt","value":"1699073045818"},{"key":"flash","value":"0"},{"key":"frm","value":"0"},{"key":"u_tz","value":"480"},{"key":"u_his","value":"11"},{"key":"u_h","value":"1080"},{"key":"u_w","value":"1920"},{"key":"u_ah","value":"1040"},{"key":"u_aw","value":"1920"},{"key":"u_cd","value":"24"},{"key":"bc","value":"31"},{"key":"bih","value":"967"},{"key":"biw","value":"636"},{"key":"brdim","value":"0,0,0,0,1920,0,1920,1040,653,967"},{"key":"vis","value":"1"},{"key":"wgl","value":"true"},{"key":"ca_type","value":"image"}],"bid":"ANyPxKrHw34YdxEP5k3AJjNpHC9eHL5evA_qa92L_JscwExeNg_Gp1Wsa7cJ8rgEPO-vGt6lbnXUMlabNQjKo_hTGVBOa7CPBQ"}},"continuation":"'+str(initdata_config["token"])+'"}'
        while 1:
            try:
                resdetail = requests.post(base_api, headers=jheaders, data=data, verify=False).json()
                time.sleep(1)
                break
            except Exception as e:
                print(f"error:{e}")
                time.sleep(1)
        # time.sleep(1)
        #'appendContinuationItemsAction'
        #'reloadContinuationItemsCommand'
        try:
            result_list = resdetail.get('onResponseReceivedActions')[-1].get('reloadContinuationItemsCommand').get(
                "continuationItems")
        except:
            result_list = resdetail.get('onResponseReceivedActions')[-1].get('appendContinuationItemsAction').get(
                "continuationItems")
        try:
            handle_list_vinfo(result_list,keyword)
        except :
            continue
        try:
            initdata_config["params"] = result_list[-1].get("continuationItemRenderer").get("continuationEndpoint").get(
                "clickTrackingParams")
        except Exception as e:
            print(f'暂无下一页：exit:{e}')
            break
        if len(c_c) >= maxvinfo:
            print(f">>>超出maxinfo限制exit")
            break
        initdata_config["token"] = result_list[-1].get("continuationItemRenderer").get("continuationEndpoint").get("continuationCommand").get("token")
        print(f'下一页游标：{initdata_config}')






if  __name__ == "__main__":
    ##走vpn节点
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    getvinfo_by_key(keyword='@ChinaYummy',maxvinfo=200000)
    getvinfo_by_key(keyword='@Sanxingdui',maxvinfo=200000)
    pd_ = pandas.DataFrame(c_c)
    pd_.to_excel(f"用户视频列表.xlsx")
