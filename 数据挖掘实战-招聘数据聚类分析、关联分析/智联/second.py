import json
import random
import threading
import time
from multiprocessing.dummy import Pool

import copyheaders
import redis
import requests
import re
pool = Pool(4)
from lxml import etree

headers = copyheaders.headers_raw_to_dict(b"""
accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
accept-language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6
sec-ch-ua: "Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"
sec-ch-ua-mobile: ?1
sec-ch-ua-platform: "Android"
sec-fetch-dest: document
sec-fetch-mode: navigate
sec-fetch-site: none
sec-fetch-user: ?1
upgrade-insecure-requests: 1
user-agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Mobile Safari/537.36 Edg/110.0.1587.46
""")


proxy_host = 'http-dynamic.xiaoxiangdaili.com'
proxy_port = 10030
proxy_username = '916959556566142976'
proxy_pwd = 'P4oVN0Lc'

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


def geturldata(data):
    url = data["页面网址"]
    while True:
        try:
            changevurl = getcurl(url)
            print(f"访问：{changevurl}")
            r = requests.get(changevurl, headers=headers, proxies=proxies, timeout=(3,5))
            if 'setCookie("acw_sc__v2", x)' in r.text:
                print(f">>> cookie失效")
                continue

            if '<title>æ»å¨éªè¯é¡µé¢</title>' in r.text:
                print(f'>>> ip失效')
                continue
            break
        except  Exception as e:
            print(f"网络异常：{e}")
            time.sleep(2)
    etree_html = etree.HTML(r.content.decode('utf-8',errors='ignore'))
    infodata = etree_html.xpath(".//div[@class='position-detail__desc-content']//text()")
    data["岗位描述"] = ''.join(infodata)
    data["工作地点"] = ''.join(etree_html.xpath("./span[@class='position-detail__address-content']//text()")).strip()
    data["职位福利"] = '|'.join([i.strip() for i in etree_html.xpath(".//div[@class='tag-list']/span//text()")])
    print(url,data["工作地点"],data["职位福利"],data["岗位描述"][:80])
    rinset.hset("zhilian-detail",data.get('页面网址'),json.dumps(data))

def getcurl(url):
    pid = url.split("/")[-1].split(".")[0]
    return f'https://m.zhaopin.com/jobs/{pid}.htm'

def start_project_detail():

    pool.map(downloaddetail,[json.loads(i.decode()) for i in rinset.lrange('plzhilian_url',0,-1)])

def downloaddetail(dataurl):

    if rinset.hexists("zhilian-detail",dataurl.get("页面网址")):
        print(f">>>已下载：{dataurl.get('页面网址')}")
        return


    try:
        geturldata(dataurl)

    except Exception as e:

        print(f"submit :{e}")



if  __name__ == "__main__":
    rinset = redis.Redis(db=3)
    start_project_detail()

