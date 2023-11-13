import json
import time
import copyheaders
import pandas
import requests

headers = copyheaders.headers_raw_to_dict(b"""
Accept:application/json, text/plain, */*
Cookie:buvid3=6016045C-99CD-44FA-6BB4-F2074A2C7AE915661infoc; i-wanna-go-back=-1; b_ut=7; _uuid=E8B8A98A-11A2-6675-4810C-7EB5E61010D52915739infoc; FEED_LIVE_VERSION=V8; buvid4=C9957598-D8C8-2CA2-9E2F-61CE00C1C56017326-023060921-JMENf0jNyDMzOjg%2BoeHFcg%3D%3D; b_nut=1686315817; header_theme_version=CLOSE; rpdid=|(YuJ~JkYYk0J'uY))m)||R); buvid_fp_plain=undefined; DedeUserID=68451030; DedeUserID__ckMd5=e48931738cbc5eac; fingerprint=e85fa83a7b467862538a31c9785ac394; buvid_fp=e85fa83a7b467862538a31c9785ac394; PVID=1; home_feed_column=5; LIVE_BUVID=AUTO2816979625007552; enable_web_push=DISABLE; bp_video_offset_68451030=857840887564075046; is-2022-channel=1; CURRENT_BLACKGAP=0; CURRENT_FNVAL=4048; browser_resolution=1872-966; innersign=0; b_lsid=8DF10BDBD_18B8F6FCC49; SESSDATA=49d9a92b%2C1714470554%2C23af9%2Ab1CjApdiPf29ARv5-NRTm8Ro0HjvU_QOsxQfMoiny5Nqyh5cTIMtKvPSHiIJeQNCZH7_ISVnVsNzZhemxkNXhWSzVNcURycFkzTFhFTWhsTUtiYm1XV2hnOHh3S0VEbU9SRGtBOHcwbTJ6ZEZ2aEpDT0I1SzlYZXhpX0hESlhaRkxhSFlXMThlV2FRIIEC; bili_jct=4a4c9c30c3d6efda5aebeb91a5fb2e68; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2OTkxNzc3NjYsImlhdCI6MTY5ODkxODUwNiwicGx0IjotMX0.L6xtoP91jhiTU3ptEEniizWQ5P5gwBoC7fsBqnxG0ns; bili_ticket_expires=1699177706; sid=6cil7c5l
Origin:https://search.bilibili.com
Referer:https://search.bilibili.com/all?keyword=%E9%98%BF%E5%87%A1%E8%BE%BE2%EF%BC%9A%E6%B0%B4%E4%B9%8B%E9%81%93&from_source=webtop_search&spm_id_from=333.1007&search_source=5&order=click&page=3&o=48
Sec-Ch-Ua:"Not/A)Brand";v="99", "Microsoft Edge";v="115", "Chromium";v="115"
Sec-Ch-Ua-Mobile:?0
Sec-Ch-Ua-Platform:"Windows"
Sec-Fetch-Dest:empty
Sec-Fetch-Mode:cors
Sec-Fetch-Site:same-site
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.203
""")




def send_get(url,headers,params):
    print(f"》》》正在访问：{url}")
    while 1:
        try:
            res = requests.get(
                url,
                timeout=(4, 5),
                headers=headers,
                params=params
            )
            time.sleep(.6)
            return res.json()
        except Exception as e:
            print(f"some error:{e}")
            time.sleep(1)


# 随机字符串


def get_detail(record):
    vname = record.get("电影名")
    for page in range(1,2):
        url = "https://api.bilibili.com/x/web-interface/wbi/search/type"
        params = {
            "category_id": "",
            "search_type": "video",
            "ad_resource": "5654",
            "__refresh__": "true",
            "_extra": "",
            "context": "",
            "page": page,
            "page_size": "42",
            "order": "click",
            "from_source": "",
            "from_spmid": "333.337",
            "platform": "pc",
            "highlight": "1",
            "single_column": "0",
            "keyword": vname,
            "qv_id": "anwYxPvzp2gNOmFXUSxqU6158eLbQwbW",
            "source_tag": "3",
            "gaia_vtoken": "",
            "dynamic_offset": "48",
            "web_location": "1430654",
            "w_rid": "c314fe1f6a600666e474944411a78a5b",
            "wts": "1698918575"
        }
        response = send_get(url,headers,params)
        comments = response.get("data",{}).get("result",[])
        for item in comments:
            try:
                saveitem = record.copy()
                saveitem["视频作者"] = item.get("author")
                saveitem["弹幕数"] = item.get("danmaku")
                saveitem["视频类型"] = item.get("typename")
                saveitem["视频标签"] = item.get("tag")
                saveitem["视频标题"] = item.get("title")
                saveitem["播放量"] =item.get("play")
                saveitem["发布时间"] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime( item.get("pubdate")))
                saveitem["点赞数"] = item.get("like")
                saveitem["视频id"] = item.get("id")
                saveitem["视频bvid"] = item.get("bvid")
                saveitem["视频aid"] = item.get("aid")
                saveitem["视频mid"] = item.get("mid")
                print(config_number["p"], page, saveitem)
                with open("video.txt", 'a', encoding='utf-8') as f:
                    f.write(json.dumps(saveitem))
                    f.write('\n')
            except Exception as e:
                print(f"parse error:{e}")

        if len(comments) < 20:
            break


# except Exception as e:
#     print(f"parse error:{e}")
#     time.sleep(1)


if __name__ == "__main__":
    records = pandas.read_excel("电影表.xlsx", dtype=str).to_dict(orient='records')
    config_number = {"p":0}
    for record in records:
        config_number["p"]+=1
        get_detail(record=record)

    with open("video.txt",'r',encoding='utf-8') as f:
        comments = [json.loads(i.strip()) for i in f.readlines()]
    df = pandas.DataFrame(comments)
    df.to_excel("视频信息.xlsx",index=False)