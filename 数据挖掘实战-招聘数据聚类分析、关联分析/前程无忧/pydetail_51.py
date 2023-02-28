import json
import time
from multiprocessing.dummy import Pool
import redis
import requests
import re

from lxml import etree

pool = Pool(6)



base_headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Host': 'msearch.51job.com',
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Mobile Safari/537.36 Edg/109.0.1518.78'
}
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



def get_hexxor(s1, _0x4e08d8):
    _0x5a5d3b = ''

    for i in range(len(s1)):
        if i % 2 != 0: continue
        _0x401af1 = int(s1[i: i + 2], 16)
        _0x105f59 = int(_0x4e08d8[i: i + 2], 16)
        _0x189e2c_10 = (_0x401af1 ^ _0x105f59)
        _0x189e2c = hex(_0x189e2c_10)[2:]
        if len(_0x189e2c) == 1:
            _0x189e2c = '0' + _0x189e2c
        _0x5a5d3b += _0x189e2c
    return _0x5a5d3b


def get_unsbox(arg1):
    _0x4b082b = [0xf, 0x23, 0x1d, 0x18, 0x21, 0x10, 0x1, 0x26, 0xa, 0x9, 0x13, 0x1f, 0x28, 0x1b, 0x16, 0x17, 0x19,
                 0xd,
                 0x6, 0xb, 0x27, 0x12, 0x14, 0x8, 0xe, 0x15, 0x20, 0x1a, 0x2, 0x1e, 0x7, 0x4, 0x11, 0x5, 0x3, 0x1c,
                 0x22, 0x25, 0xc, 0x24]
    _0x4da0dc = []
    _0x12605e = ''
    for i in _0x4b082b:
        _0x4da0dc.append(arg1[i - 1])
    _0x12605e = "".join(_0x4da0dc)
    return _0x12605e

def setheaders():
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'Host': 'jobs.51job.com',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:2.0b4pre) Gecko/20100815 Minefield/4.0b4pre',
    }
    while True:
        try:
            r = requests.get('https://jobs.51job.com/beijing-dxq/134663856.html', headers=headers
                             , proxies=proxies,timeout=(3,5),verify=False)
            arg1s = re.findall("arg1=\'(.*?)\'", r.text)
            if len(arg1s) == 0:
                print(f"setcookie {arg1s} ")
                time.sleep(4)
                continue
            break
        except  Exception as e:
            print(f"net error：{e}")
    s1 = get_unsbox(arg1s[0])
    _0x4e08d8 = "3000176000856006061501533003690027800375"
    _0x12605e = get_hexxor(s1, _0x4e08d8)
    print(f"更新_0x12605e： {_0x12605e}")
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'Host': 'jobs.51job.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:2.0b4pre) Gecko/20100815 Minefield/4.0b4pre',
        'X-Requested-With': 'XMLHttpRequest',
        "cookie": "acw_sc__v2=%s" % _0x12605e
    }

def geturldata(sjob):


    try:

        api = sjob.get("url")

        res = requests.get(api, headers=base_headers, proxies=proxies)

        if '滑动验证页面' in res.content.decode():
            print('滑动验证页面')
        else:
            document_etree = etree.HTML(res.content.decode())
            jobinfoDesModule = ''.join(document_etree.xpath(".//article[@class='jobinfoDesModule']//text()")).strip()
            print(sjob["岗位名称"], jobinfoDesModule[:80])
            sjob["全文"] = jobinfoDesModule
            redis_con.hset("jobdetail", sjob.get("url"), json.dumps(sjob))



    except Exception as e:


        print(f"submit network error: {e}")







def start_project_detail():
    job_list = redis_con.hvals("joblist")
    pool.map(downloaddetail,job_list)
    # for ijob in job_list:
    #     downloaddetail(ijob)


def downloaddetail(jobSome):
    rpop = jobSome.decode()
    item = json.loads(rpop)
    job_item = item.copy()

    if redis_con.hexists("jobdetail",job_item.get("url")):
        print(f">>>已下载：{job_item.get('url')}")
        return
    geturldata(job_item)


if  __name__ == "__main__":
    print(proxyMeta)
    redis_con = redis.Redis(db=3)
    start_project_detail()

