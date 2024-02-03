import hashlib
import json
import pprint
import time

import copyheaders
import pandas
import requests
from lxml import etree

headers = copyheaders.headers_raw_to_dict(b"""
Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Cookie:uuid_n_v=v1; uuid=6D7BA320BF6711EE9B05F94429EFE3ED2D6CE98ED21747D689B16927A6B18AEB; _csrf=d76242a5e39bdc16dec6a95bdbba57b512d5af3dc641dc4682e5b2bab4d3c8be; Hm_lvt_703e94591e87be68cc8da0da7cbd0be2=1706616144; _lxsdk_cuid=1896db100fec8-01656acf45c0ac-7e565474-1fa400-1896db100fec8; __mta=150197897.1706616147052.1706631375583.1706631752976.6; featrues=[object Object]; _lxsdk=6D7BA320BF6711EE9B05F94429EFE3ED2D6CE98ED21747D689B16927A6B18AEB; _lxsdk_s=18d5b18cae9-639-6cc-a39%7C%7C125; Hm_lpvt_703e94591e87be68cc8da0da7cbd0be2=1706634853
Referer:https://www.maoyan.com/films?yearId=7&showType=3&offset=450
Host: www.maoyan.com
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0
""")


def send_get(url, heades, params):
    while 1:
        try:
            response = requests.get(
                url,
                headers=heades,
                params=params,
                timeout=(4, 5),
            )
            return response.text
        except Exception as e:
            print(f"some error:{e}")
            time.sleep(10)


def search_movie(irecord):
    movieid = irecord.get("id")
    print(f"访问；{movieid}")
    url = f"https://www.maoyan.com/ajax/films/{movieid}"
    params = {
        "timeStamp": "1706634852950",
        "index": "8",
        "signKey": "1e95087592bc9640484dd8ec68f06211",
        "channelId": "40011",
        "sVersion": "1",
        "webdriver": "false"
    }
    response = send_get(url, headers, params=params)
    document = etree.HTML(response)
    saveitem = {}
    saveitem["电影名"] = "".join(document.xpath(".//div[@class='movie-brief-container']/h1//text()"))
    saveitem["电影英文名"] = "".join(document.xpath(".//div[@class='movie-brief-container']/div//text()"))
    saveitem["电影id"] = movieid
    saveitem["封面图"] = "".join(document.xpath(".//div[@class='avatar-shadow']/img/@src"))
    saveitem["电影简介"] = "".join(document.xpath(".//div[@class='mod-content']/span[@class='dra']//text()")).strip()
    saveitem["评分"] = ""
    saveitem["评论人数"] = ""
    saveitem["出品国家"] = ""
    saveitem["电影语言"] = ""
    saveitem["电影类别"] = ""
    saveitem["上映时间"] = ""
    saveitem["片长"] = ""
    saveitem["tag描述"] = ""
    saveitem["短评数"] = ""
    saveitem["影评数"] = ""
    saveitem["推荐原因"] = ""
    saveitem["演职人员"] = ""
    saveitem["年份"] = ""

    for itab in document.xpath(".//div[@class='celebrity-container']/div[@class='celebrity-group']"):
        title = "".join(itab.xpath(".//div[@class='celebrity-type']//text()")).strip()
        value_ = "; ".join(itab.xpath(".//ul[@class='celebrity-list clearfix']//div[@class='name']/text()"))
        c_value = f'{title}：{value_}\n'
        saveitem["演职人员"] += c_value
    # wish_api = f'https://m.douban.com/rexxar/api/v2/movie/{movieid}/rating?ck=0dIg&for_mobile=1'
    # wish_response = send_get(wish_api, headers, {})
    # saveitem["观看人数"] = wish_response.get("done_count")
    # saveitem["想看人数"] = wish_response.get("wish_count")

    pprint.pprint(saveitem)


if __name__ == "__main__":
    search_movie({
        "id": '1211270'
    })
