import json
import time

import copyheaders
import redis
import requests
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


headers= copyheaders.headers_raw_to_dict(b"""
Accept: application/json, text/plain, */*
Accept-Encoding: gzip, deflate, br
Accept-Language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6
Connection: keep-alive
Content-Length: 500
Content-Type: application/json;charset=UTF-8;
Cookie: __uuid=1669357599496.89; __gc_id=ea0d0b8b7c6c479799a9e2953ef95cd2; _gcl_au=1.1.323246243.1669357604; need_bind_tel=false; new_user=false; c_flag=fce42dc90d3716d4e99101669f003c9d; imClientId=58640ed5573c62ae71b497382f997a51; imId=58640ed5573c62ae824c495b984a44e4; imClientId_0=58640ed5573c62ae71b497382f997a51; imId_0=58640ed5573c62ae824c495b984a44e4; XSRF-TOKEN=yd8dU_4ETdOSgK8RHobv9Q; __tlog=1676696046164.60%7C00000000%7C00000000%7Cs_o_007%7Cs_o_007; Hm_lvt_a2647413544f5a04f00da7eee0d5e200=1676696047; acw_tc=2760829816766960482627723e792f5eee3bdc69141f76fa5278b2826726fc; Hm_lpvt_a2647413544f5a04f00da7eee0d5e200=1676696279; __session_seq=7; __uv_seq=7
Host: apic.liepin.com
Origin: https://www.liepin.com
Referer: https://www.liepin.com/
sec-ch-ua: "Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "Windows"
Sec-Fetch-Dest: empty
Sec-Fetch-Mode: cors
Sec-Fetch-Site: same-site
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.46
X-Client-Type: web
X-Fscp-Bi-Stat: {"location": "https://www.liepin.com/zhaopin/?city=410&dq=410&pubTime=&currentPage=0&pageSize=40&key=java&suggestTag=&workYearCode=0&compId=&compName=&compTag=&industry=H04$H04&salary=&jobKind=&compScale=&compKind=&compStage=&eduLevel=&otherCity=&scene=condition&suggestId="}
X-Fscp-Fe-Version: 214116d
X-Fscp-Std-Info: {"client_id": "40108"}
X-Fscp-Trace-Id: 9e4059e2-292e-46f6-af3b-7c4a0d7dfb32
X-Fscp-Version: 1.1
X-Requested-With: XMLHttpRequest
X-XSRF-TOKEN: yd8dU_4ETdOSgK8RHobv9Q
""")

def load_citys():
    citys = []
    with open("city.json",'r',encoding='utf-8') as f:
        org = json.loads(f.read())
        cs = org.values()
        for ics in cs:
            if ics.get("level") in ['3','7']:
                citys.append((
                    ics.get("code"),
                    ics.get("name")
                ))
    return citys
def submit_data(data):
    while 1:
        try:
            res =requests.post(
                'https://apic.liepin.com/api/com.liepin.searchfront4c.pc-search-job',
                headers=headers,
                data=data,
                timeout=(4,5),proxies=proxies
            )
            return res.json()
        except Exception as e:
            print(f">>> ero:{e}")
            time.sleep(1)

def download_city(keyword):
    for page in range(1,3):
        data = {"data":{"mainSearchPcConditionForm":{"city":str(citycode),"dq":str(citycode),"pubTime":"","currentPage":page,"pageSize":40,"key":keyword,"suggestTag":"","workYearCode":"0","compId":"","compName":"","compTag":"","industry":"H04$H04","salary":"","jobKind":"","compScale":"","compKind":"","compStage":"","eduLevel":"","otherCity":""},"passThroughForm":{"ckId":"xp8hh6cwnt1aobcxor4ui32fscms4gz8","scene":"page","skId":"ijkcdpxmnjfsesbk1qjv9iyq489d4p48","fkId":"ijkcdpxmnjfsesbk1qjv9iyq489d4p48","sfrom":"search_job_pc"}}}
        res  = submit_data(json.dumps(data))
        jobCardList = res.get("data").get("data").get("jobCardList")
        for job in jobCardList:
            saveinfo = {}
            saveinfo["地区"] = job.get("job").get("dq")
            saveinfo["url"] = job.get("job").get("link")
            saveinfo["发布时间"] = job.get("job").get("refreshTime")
            saveinfo["学历要求"] = job.get("job").get("requireEduLevel")
            saveinfo["经验要求"] = job.get("job").get("requireWorkYears")
            saveinfo["标题"] = job.get("job").get("title")
            saveinfo["薪资"] = job.get("job").get("salary")
            saveinfo["岗位属性"] = "|".join(job.get("job").get("labels"))
            saveinfo["公司性质"] = job.get("comp").get("compIndustry")
            saveinfo["公司名称"] = job.get("comp").get("compName")
            saveinfo["公司规模"] = job.get("comp").get("compScale")
            print(cityname,keyword,page,saveinfo)
            __sql.lpush("liepin-url",json.dumps(saveinfo))
        if len(jobCardList) <= 10:
            break

if  __name__ == "__main__":

    #信托、证券、投资、理财、保险顾问
    __sql = redis.Redis(db=3)
    citys = load_citys()
    print(citys)
    for keyword in ['信托','证券','投资','理财','保险顾问']:
        for citycode, cityname in citys:
            try:
                download_city(keyword=keyword)  # 投资
            except Exception as e:
                print(f">>> error:{e}")
