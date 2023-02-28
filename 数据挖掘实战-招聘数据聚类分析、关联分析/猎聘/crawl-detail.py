import json
import time
from multiprocessing.dummy import Pool

import copyheaders
import redis
import requests
import re

from lxml import etree

pool = Pool(6)



base_headers = copyheaders.headers_raw_to_dict(b"""
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Host: www.liepin.com
Referer: https://safe.liepin.com/
sec-ch-ua: "Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "Windows"
Sec-Fetch-Dest: document
Sec-Fetch-Mode: navigate
Sec-Fetch-Site: same-site
Sec-Fetch-User: ?1
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.46
""")
proxy_host = 'http-dynamic.xiaoxiangdaili.com'
proxy_port = 10030
proxy_username = '940265137389326336'
proxy_pwd = 'GIv65kKc'

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





def geturldata(sjob):


    try:

            api = sjob.get("url")

            while 1:
                try:
                    res = requests.get(api, headers=base_headers, proxies=proxies,timeout=(4,5))
                    break
                except Exception as e:
                    print(f">>> error:{e}")
                    time.sleep(1)



            document_etree = etree.HTML(res.content.decode())
            jobinfoDesModule = ''.join(document_etree.xpath(".//dd[@data-selector='job-intro-content']//text()")).strip()
            print(sjob["标题"], jobinfoDesModule[:80])
            sjob["全文"] = jobinfoDesModule
            redis_con.hset("liepin-jobdetail", sjob.get("url"), json.dumps(sjob))



    except Exception as e:


        print(f"submit network error: {e}")







def start_project_detail():
    job_list = [json.loads(i.decode()) for i in redis_con.lrange("liepin-url",0,-1)]
    pool.map(downloaddetail,job_list)


def downloaddetail(item):
    job_item = item.copy()

    if redis_con.hexists("liepin-jobdetail",job_item.get("url")):
        print(f">>>已下载：{job_item.get('url')}")
        return
    geturldata(job_item)


if  __name__ == "__main__":
    print(proxyMeta)
    redis_con = redis.Redis(db=3)
    start_project_detail()

