import time

import copyheaders
import pymongo
import requests
from lxml import etree

headers = copyheaders.headers_raw_to_dict(b"""
accept: */*
accept-encoding: gzip, deflate, br
accept-language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6
referer: https://search.jd.com/Search?keyword=%E5%8D%8E%E4%B8%BA%E6%97%97%E8%88%B0%E5%BA%97&qrst=1&wq=%E5%8D%8E%E4%B8%BA%E6%97%97%E8%88%B0%E5%BA%97&ev=exbrand_%E5%8D%8E%E4%B8%BA%EF%BC%88HUAWEI%EF%BC%89%5E&pvid=ab539e5ef8c8440aa3eddcaf9bd93243&page=3&s=58&click=0
sec-ch-ua: "Chromium";v="112", "Microsoft Edge";v="112", "Not:A-Brand";v="99"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "Windows"
sec-fetch-dest: empty
sec-fetch-mode: cors
sec-fetch-site: same-origin
user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.58
x-requested-with: XMLHttpRequest
cookie: __jda=122270672.1682398325144483666592.1682398325.1682398325.1682398325.1; __jdc=122270672; __jdv=122270672|direct|-|none|-|1682398325144; 3AB9D23F7A4B3CSS=jdd03MHKJK4Y46FAF27B5EWZ5MMYBGJWSPHYJKVTBCZCEPD7IVZ3V33ZWQCAGP7FZ6YS4DQM32SPSGDIEDWZNP62Z76IGPYAAAAMHW3ATS7AAAAAAC6ICROBDNSSF7YX; jsavif=0; shshshfpb=trzLaXD38vCGKH1XxoMIL8g; shshshfp=c20285500712d78efe0d8a5d4fd8efb1; shshshfpa=178af350-5338-b9da-c676-83572aaa0cc8-1653288709; shshshfpx=178af350-5338-b9da-c676-83572aaa0cc8-1653288709; __jdu=1682398325144483666592; areaId=18; ipLoc-djd=18-1495-1498-30785; 3AB9D23F7A4B3C9B=MHKJK4Y46FAF27B5EWZ5MMYBGJWSPHYJKVTBCZCEPD7IVZ3V33ZWQCAGP7FZ6YS4DQM32SPSGDIEDWZNP62Z76IGPY; wlfstk_smdl=pixp3vz264mpv6u0u76ko2m7fr7m51yu; TrackID=1RCJqAYzn_o6KxwenvMnflRsO1VSvsOyRAQVUZh73nSRma2KsvL9o92ksjCfW-hRHoPjoKsctqkh-QMMbP8szajVjOYj-Wz-d9_IRVIG5yDb2S5L1mgAV9LcvnqllpYhn; thor=BC33C8EA7E363B98D12BB378AAB2F5B38462C48CB77B7DC35C45F6E9A70DB32E0B8E1F1C40EE2AFA6B8C24653EBD72874EA6F416A59D813AD390EF7618760D94017F352C7C5473826A4D36FFF886C5D1DD11D988132A012DB1C1225A2A09C28A1BF0159029B2A0FC3D51C8B233562E90398BB3FE0857D6FF3365816FF9C771EC626DCD1FD0FEFC4543102081171F5A72527EEE3B837A671A4C779D8D7E2B493D; pinId=bb9Gfy_M8--7xng6PKS6wrV9-x-f3wj7; pin=jd_6690177c46b65; unick=%E4%BB%98%E7%A5%A5%E5%8F%8B; ceshi3.com=201; _tp=o8gWPCPC7w7OIx5RrTeOrZaSLHhWqEdfCKJbzIaLQFY%3D; _pst=jd_6690177c46b65; token=e232226197ed44f65fe533df1458e427,3,934665; __tk=k1O3BbfxAjBYlzGJTyCOZlyJSjClBc5gjzkHYzRuRjklklzVkyBHBbfkSjCHZm35SlVQqczd,3,934665; __jdb=122270672.7.1682398325144483666592|1.1682398325; shshshsID=0bb05040deba686b2ccd9b8977512dcb_3_1682398409297
""")

json_headers = copyheaders.headers_raw_to_dict(b"""
accept: application/json, text/javascript, */*; q=0.01
accept-language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6
content-type: application/json;charset=gbk
cookie: __jda=122270672.1682398325144483666592.1682398325.1682398325.1682398325.1; __jdc=122270672; __jdv=122270672|direct|-|none|-|1682398325144; 3AB9D23F7A4B3CSS=jdd03MHKJK4Y46FAF27B5EWZ5MMYBGJWSPHYJKVTBCZCEPD7IVZ3V33ZWQCAGP7FZ6YS4DQM32SPSGDIEDWZNP62Z76IGPYAAAAMHW3ATS7AAAAAAC6ICROBDNSSF7YX; jsavif=0; shshshfpb=trzLaXD38vCGKH1XxoMIL8g; shshshfp=c20285500712d78efe0d8a5d4fd8efb1; shshshfpa=178af350-5338-b9da-c676-83572aaa0cc8-1653288709; shshshfpx=178af350-5338-b9da-c676-83572aaa0cc8-1653288709; __jdu=1682398325144483666592; areaId=18; ipLoc-djd=18-1495-1498-30785; 3AB9D23F7A4B3C9B=MHKJK4Y46FAF27B5EWZ5MMYBGJWSPHYJKVTBCZCEPD7IVZ3V33ZWQCAGP7FZ6YS4DQM32SPSGDIEDWZNP62Z76IGPY; wlfstk_smdl=pixp3vz264mpv6u0u76ko2m7fr7m51yu; TrackID=1RCJqAYzn_o6KxwenvMnflRsO1VSvsOyRAQVUZh73nSRma2KsvL9o92ksjCfW-hRHoPjoKsctqkh-QMMbP8szajVjOYj-Wz-d9_IRVIG5yDb2S5L1mgAV9LcvnqllpYhn; thor=BC33C8EA7E363B98D12BB378AAB2F5B38462C48CB77B7DC35C45F6E9A70DB32E0B8E1F1C40EE2AFA6B8C24653EBD72874EA6F416A59D813AD390EF7618760D94017F352C7C5473826A4D36FFF886C5D1DD11D988132A012DB1C1225A2A09C28A1BF0159029B2A0FC3D51C8B233562E90398BB3FE0857D6FF3365816FF9C771EC626DCD1FD0FEFC4543102081171F5A72527EEE3B837A671A4C779D8D7E2B493D; pinId=bb9Gfy_M8--7xng6PKS6wrV9-x-f3wj7; pin=jd_6690177c46b65; unick=%E4%BB%98%E7%A5%A5%E5%8F%8B; ceshi3.com=201; _tp=o8gWPCPC7w7OIx5RrTeOrZaSLHhWqEdfCKJbzIaLQFY%3D; _pst=jd_6690177c46b65; token=e232226197ed44f65fe533df1458e427,3,934665; __tk=k1O3BbfxAjBYlzGJTyCOZlyJSjClBc5gjzkHYzRuRjklklzVkyBHBbfkSjCHZm35SlVQqczd,3,934665; __jdb=122270672.7.1682398325144483666592|1.1682398325; shshshsID=0bb05040deba686b2ccd9b8977512dcb_3_1682398409297
origin: https://item.jd.com
referer: https://item.jd.com/
sec-ch-ua: "Chromium";v="112", "Microsoft Edge";v="112", "Not:A-Brand";v="99"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "Windows"
sec-fetch-dest: empty
sec-fetch-mode: cors
sec-fetch-site: same-site
user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.58
x-referer-page: https://item.jd.com/100035295081.html
x-rp-client: h5_1.0.0
""")

def get_shop_detail(api,params):
    while 1:
        try:
            res = requests.get(
                api,
                headers=headers,
                timeout=(3,4),
                params=params
            )
            time.sleep(1.4)
            return res.text
        except Exception as e:
            print(f"error:{e}")
            time.sleep(1)

def search_shop():
    api = 'https://search.jd.com/s_new.php'

    for page in range(1,17):

        params = {
            "keyword":'小米京东自营官方旗舰店',
            "qrst":'1',
            "wq":'小米京东自营官方旗舰店',
            "ev":'exbrand_小米（MI）^',
            "pvid":'800aea2bbafd4f5dac19c8164ae70c4b',
            "page":page,
            "s":58,
            "click":0
        }

        res  = get_shop_detail(api,params)

        document = etree.HTML(res)

        uls = document.xpath(".//ul[@class='gl-warp clearfix']/li")

        for iurl in uls:
            try:
                saveitem = {}
                saveitem["skuid"] = ''.join(iurl.xpath("./@data-sku"))
                saveitem["price"] = ''.join(iurl.xpath(".//div[@class='p-price']//text()")).strip()
                saveitem["cover"] = 'https:' + ''.join(
                    iurl.xpath(".//div[@class='p-img']/a/img/@data-lazy-img")).strip()
                saveitem["title"] = ''.join(iurl.xpath(".//div[@class='p-name p-name-type-2']//text()")).strip()
                saveitem["_id"] = saveitem["skuid"]
                print(saveitem)
                database["shop"].insert_one(saveitem)
            except Exception as e:
                print(f"error:{e}")
        if len(uls) < 10:
            break


def search_detail():
    shops = database["shop"].find({})
    for ishop in shops:
        api = f'https://item.jd.com/{ishop["skuid"]}.html#product-detail'
        response = get_shop_detail(api,params={})
        document = etree.HTML(response)

        part = document.xpath(".//ul[@class='parameter2 p-parameter-list']/li")
        saveitem = ishop.copy()
        saveitem["_id"] = ishop["skuid"]
        saveitem["商品名称"] = ""
        saveitem["商品编号"] = ""
        saveitem["商品毛重"] = ""
        saveitem["商品产地"] = ""
        saveitem["CPU型号"] = ""
        saveitem["运行内存"] = ""
        saveitem["机身颜色"] = ""
        saveitem["三防标准"] = ""
        saveitem["屏幕分辨率"] = ""
        saveitem["充电功率"] = ""
        saveitem["机身内存"] = ""
        saveitem["风格"] = ""
        saveitem["屏幕材质"] = ""
        saveitem["后摄主像素"] = ""
        saveitem["机身色系"] = ""
        for ipart in part:
            values = ipart.xpath("string(.)")
            if '商品名称' in values:
                saveitem["商品名称"] = values.split("：")[-1]
                continue
            if '商品编号' in values:
                saveitem["商品编号"] = values.split("：")[-1]
                continue

            if '商品毛重' in values:
                saveitem["商品毛重"] = values.split("：")[-1]
                continue

            if '商品产地' in values:
                saveitem["商品产地"] = values.split("：")[-1]
                continue

            if 'CPU型号' in values:
                saveitem["CPU型号"] = values.split("：")[-1]
                continue

            if '运行内存' in values:
                saveitem["运行内存"] = values.split("：")[-1]
                continue

            if '机身颜色' in values:
                saveitem["机身颜色"] = values.split("：")[-1]
                continue

            if '三防标准' in values:
                saveitem["三防标准"] = values.split("：")[-1]
                continue

            if '屏幕分辨率' in values:
                saveitem["屏幕分辨率"] = values.split("：")[-1]
                continue

            if '充电功率' in values:
                saveitem["充电功率"] = values.split("：")[-1]
                continue

            if '机身内存' in values:
                saveitem["机身内存"] = values.split("：")[-1]
                continue
            if '风格' in values:
                saveitem["风格"] = values.split("：")[-1]
                continue

            if '屏幕材质' in values:
                saveitem["屏幕材质"] = values.split("：")[-1]
                continue

            if '后摄主像素' in values:
                saveitem["后摄主像素"] = values.split("：")[-1]
                continue
            if '机身色系' in values:
                saveitem["机身色系"] = values.split("：")[-1]
                continue
        print(saveitem)

        try:
            database["shopdetail"].insert_one(saveitem)
        except Exception as e:
            print(f"error:{e}")


def search_comment():
    shops = database["shop"].find({})
    for ishop in shops:

        for mlabel,mid in [('好评',3),('中评',2),('差评',1)]:
            for page in range(1,20):
                try:
                    api = 'https://api.m.jd.com/'
                    params = {
                        "appid": "item-v3",
                        "functionId": 'pc_club_skuProductPageComments',
                        "client": 'pc',
                        "clientVersion": '1.0.0',
                        "t": '1682398783121',
                        "loginType": 3,
                        "uuid": '122270672.1682398325144483666592.1682398325.1682398325.1682398325.1',
                        "productId": ishop["skuid"],
                        "score": mid,
                        "sortType": 5,
                        "page": page,
                        "pageSize": 10,
                        "isShadowSku": 0,
                        "rid": 0,
                        "fold": 1
                    }

                    res = requests.get(api, headers=json_headers, params=params, timeout=(4, 5)).json()
                    time.sleep(1)
                    comments = res.get("comments",[])
                    for icomment in comments:
                        saveitem = ishop.copy()
                        saveitem["_id"] = icomment.get("id")
                        saveitem["评论类型"] = mlabel
                        saveitem['评价日距离购买天数'] = icomment.get("days")
                        saveitem['评论内容'] = icomment.get("content")
                        saveitem['评论创建日期'] = icomment.get("creationTime")
                        saveitem['发布图片数'] = icomment.get("imageCount")
                        saveitem['用户名'] = icomment.get("nickname")
                        saveitem['产品颜色'] = icomment.get("productColor")
                        saveitem['产品存储'] = icomment.get("productSize")
                        saveitem['评论回复数'] = icomment.get("replyCount")
                        saveitem['评论有用数'] = icomment.get("usefulVoteCount")
                        saveitem['会员标志'] = icomment.get("plusAvailable") ##101,201是会员
                        print(saveitem)
                        try:
                            database["comment"].insert_one(saveitem)
                        except Exception as e:
                            print(f"insert error:{e}")


                    if len(comments) < 7:
                        break
                except Exception as e:
                    print(f"error:{e}")


if __name__ == "__main__":


    mongodb_database = pymongo.MongoClient()
    database = mongodb_database["xm_database"]
    # search_shop()
    # search_detail()
    search_comment()