##导入相关包
import json
import os
import random
import time
import copyheaders
import pandas
import pymongo
import requests




history_page = []


def reload_crawl_done_page():
    if not os.path.exists("./config/done.txt"):
        return
    with open("./config/done.txt",'r',encoding='utf-8') as f:
        done_pages = [i.strip() for i in f.readlines()]
    for done in done_pages:
        history_page.append(done)

##调用函数获取评论数据
def start_search(shop_id,max_page,user_login_token):
    for page in range(0,int(max_page)):

        current_page = f'{shop_id}-{page}'
        if current_page in history_page:
            print(f">>> 当前页面已经被采集过：{current_page}")
            continue

        ##评论接口
        api = 'https://m.dianping.com/ugc/review/reviewlist'

        ##请求参数：cx  _token  mtsireferer 不做校验，可以保留固定值
        params = {
            "yodaReady": "wx",
            "csecplatform": "3",
            "csecversion": "1.4.0",
            "optimus_platform": "13",
            "optimus_partner": "203",
            "optimus_risk_level": "71",
            "optimus_code": "10",
            "tagType": "1",
            "tag": "全部",
            "offset": str(page*10),
            "shopUuid": shop_id,
        }

        ##发送请求
        while 1:
            try:
                headers = {
                    'Host': 'm.dianping.com',
                    'Connection': 'keep-alive',
                    'mtgsig': '{"a1":"1.2","a2":1703609036275,"a3":"y47378326u505vv81591y895001221v181x400693v59797873w7yv24","a4":"e36c73498962e47c49736ce37ce46289ba9e6e44c67a0ccd","a5":"RTQ4z5bb26fdZ17AaVaMivKen+YsqBgYaJVzj24ost1cf7eQOVXjNFXMQUvW24uyEPPtY8GyfxYWW+r6VEVC9V1oAyHEGkxoUMi29ksbFmIeWVB5qf89Bln5cM+NsqvPK1tW9FmZvW6N1tho/cx/+8O/zJw80YC347AmWoh4XlZ9CGlc1j34mfvmIyVnt+65M1mBz6kuYBpFiG/xK79AtFIAYRbt0JUvvRGf9QY8uOzYPrh+KftNk70eRX8VHuPMb9oWJFZF59iWd0Z6R0sqJ8U4grUp2MgsRWTm1kV6tCg+lSd7ssMiAc==","a6":"w1.25HmJjjRDkffs3gIzYJdpNe8CQ0RJmpwpjmH9pRoJUEXfdSd6uUt/4rKioU2vlvgGP0BYff0qsB4PTFzIXI4CfGcZbgY11iTKkJlIbGxA1OX3AQSbjK7yxxqogsjjYN8jx+aBxBKOIUPwAGgFAobSrX26xCRvI87Q1VvvD2LAdILRdJr1HZqt/NysPhKCGq+Vmi5AemRgzRtvjW779bqRebVVLLzOsErh1vxlgywO6uYpPrjs5mFJZoGAzMDm82S3eiIfLp6+6AGYdOefNGdr/lwve2uxXLOmogPVTnBm4z8BfpT1cOV5IGQgpQ0EuWGJUVwLUm77KGJzayQoKsOuckqB73ZuLKjf6clUE22nUon4TrSn/VEEmg8gxUy29S5I/CDfA1unU6XMH2kNOG0V2eVqa8XoXNWyT+Y7l8AISZRdTYZrUVtMZWYTKpW/U4xqoDhCsEcxo2VXmikmVB+TyZVQBNO7z85qj3GO1o7yqN8=","a7":"wx734c1ad7b3562129","x0":3,"d1":"7dc765c0bc6e58a2df2e80c1f8c892b3"}',
                    'channel': 'weixin',
                    'channelversion': '3.9.8',
                    'minaversion': '9.48.1',
                    'wechatversion': '3.9.8',
                    'sdkversion': '3.2.5',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 MicroMessenger/7.0.20.1781(0x6700143B) NetType/WIFI MiniProgramEnv/Windows WindowsWechat/WMPF WindowsWechat(0x6309080f)XWEB/8501',
                    'Content-Type': 'application/json',
                    'token': user_login_token,
                    'minaname': 'dianping-wxapp',
                    'Accept': '*/*',
                    'Accept-Language': '*',
                    'Referer': 'https://servicewechat.com/wx734c1ad7b3562129/448/page-frame.html',
                }
                proxy_host = 'http-dynamic-S02.xiaoxiangdaili.com'
                proxy_port = 10030
                proxy_username = '916959556566142976'
                proxy_pwd = 'qHDFwFYk'

                proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
                    "host": proxy_host,
                    "port": proxy_port,
                    "user": proxy_username,
                    "pass": proxy_pwd,
                }

                proxies = {
                    'http': proxyMeta,
                    'https': proxyMeta,
                }
                res = requests.get(api, headers=headers, params=params,timeout=(10,10)).json()
                time.sleep(random.uniform(.2,.4))
                break
            except Exception as e:
                print(f"账号异常:{e}")
                time.sleep(10)
                return
        ##获取评论列表
        shopReviewInfo = res['reviewInfo']['reviewListInfo']["reviewList"]
        for isr in shopReviewInfo:
            comtents = []
            ##解析出评论文本
            if type(isr.get("reviewBody")) ==  str:
                continue
            else:
                reviewBody = isr.get("reviewBody").get("children")
                for irev in reviewBody:

                    try:
                        if irev.get("children") is not None:
                            for ichildren in irev.get("children"):
                                if ichildren.get("type") == 'text':
                                    comtents.append(ichildren.get("text"))
                        else:
                            if irev.get("type") == 'text':
                                comtents.append(irev.get("text"))
                    except Exception as e:
                        print(f'parse error:{e}')
                        time.sleep(1)
            comtents_ = '\t'.join(comtents)

            saveitem = {}
            saveitem["店铺id"] = shop_id
            saveitem["当前页码"] = page
            saveitem["星级"] = isr.get("accurateStar")
            saveitem["人均"] = isr.get("avgPrice")
            saveitem["评论"] = comtents_
            saveitem["图片"] = [i.get("bigurl") for i in isr.get("reviewPics")]
            saveitem["评论时间"] = isr.get("addTime")
            saveitem['用户id'] = isr.get("userId")
            saveitem['用户名'] = isr.get("userNickName")
            saveitem["点赞数"] = isr.get("flowerTotal")
            saveitem["评论数"] = isr.get("followNoteNo")
            print(shop_id,f'[{page}/{int(max_page)}]',saveitem)
            with open(f"./config/{shop_id}_data.txt",'a',encoding='utf-8') as f:
                f.write(json.dumps(saveitem))
                f.write('\n')

        with open("./config/done.txt",'a',encoding='utf-8') as f:
            f.write(current_page)
            f.write('\n')

        if len(shopReviewInfo) < 5:
            break



if  __name__ == "__main__":

    reload_crawl_done_page()

    #https://maccount.dianping.com/mlogin/smslogin?redir=https%3A%2F%2Fm.dianping.com%2Fnmy%2Fmyinfo



    # records = pandas.read_excel("./config/店铺信息采集结果筛选.xlsx",dtype=str).to_dict(orient='records')[::-1]
    # for irecord in records:
    #     shop_id = irecord.get("店铺id")
    #     max_page = float(irecord.get("评论数")) // 10 + 1
    #     start_search(shop_id=shop_id,max_page=max_page,user_login_token='499abe5e7b59e432986a12ed2b2f22de58ddf444d64bf965dd6b999fc32255adf05fba47a1c383d6698564e13d98422ed9d959004bd2934da5f823ed3e741785')


    records = pandas.read_excel("./config/店铺信息采集结果筛选.xlsx",dtype=str).to_dict(orient='records')[::-1]
    for irecord in records:
        shop_id = irecord.get("店铺id")
        max_page = 2000
        start_search(shop_id=shop_id,max_page=max_page,user_login_token='2a87051d614cb41df97ead0b17d4fed6e6fb3bda2d37cf5ac3dfa7ec0a437be5916ed7af361535043f3f33a8cf101905e7aa00bd77a1e816c55cd09d91139970')
    
    # records = pandas.read_excel("./config/店铺信息采集结果筛选.xlsx",dtype=str).to_dict(orient='records')[::-1]
    # for irecord in records:
    #     shop_id = irecord.get("店铺id")
    #     max_page = 1000
    #     start_search(shop_id=shop_id,max_page=max_page,user_login_token='b758f0794a09a79606d26658a6b8a47f7565928a76cbd78ddc33a3311a6f2e0b8914c7cb2554490a0afae761c15c140cd60e0665981d15f433b1e193161bf5ee')
