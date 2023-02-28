import json
import time
import urllib.parse
from hashlib import sha256
import hmac
import redis
import requests

BASE_HEADERS = {
    'Accept': 'application/json, text/plain, */*',
    'Host': 'cupid.51job.com',
    'Origin': 'https://we.51job.com',
    'Referer': 'https://we.51job.com/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 Edg/109.0.1518.78'
}
BASE_URL = '/open/noauth/search-pc'
BASE_HOST = 'https://cupid.51job.com'




def parse_headers(base_url):
    headers = BASE_HEADERS.copy()
    headers["sign"] = get_sign(base_url, "abfc8f9dcf8c3f3d8aa294ac5f2cf2cc7767e5592590f39c3f503271dd68562b")
    return headers

def dowoload_by_key(function,workedu,citycode):
        for page in range(1, 200):
            base_url = '/open/noauth/search-pc' \
                       '?api_key=51job&' \
                       f'timestamp={int(time.time())}&' \
                       f'keyword=&searchType=2&' \
                       f'function={function}&industry=&' \
                       f'jobArea={citycode}&jobArea2=&' \
                       'landmark=&metro=&salary=&' \
                       f'workYear=&degree={workedu}&companyType=&' \
                       'companySize=&jobType=&issueDate=&' \
                       f'sortType=0&pageNum={page}&requestId=d5c90c9dab5a6ba74373dec114e367dd&' \
                       'pageSize=50&source=1&accountId=&pageCode=sou%7Csou%7Csoulb'

            headers = parse_headers(base_url)
            request_api = BASE_HOST + base_url

            response_body = submit_req(request_api, headers)
            print(headers["sign"])
            print(request_api)
            job_items = response_body.get("resultbody", {}).get("job", {}).get("items", [])
            for ijob in job_items:
                try:
                    saveitem = {}
                    saveitem['发布时间'] = ijob.get("issueDateString")
                    saveitem['岗位名称'] = ijob.get("jobName")
                    saveitem['属性列表'] = '|'.join(ijob.get("jobTags"))
                    saveitem['公司名'] = ijob.get("companyName")
                    saveitem['公司性质'] = ijob.get("companyTypeString")
                    saveitem['公司规模'] = ijob.get("companySizeString")
                    saveitem['公司标签'] = ijob.get("industryType1Str")
                    saveitem['url'] = f'https://msearch.51job.com/jobs/company/{ijob.get("jobId")}.html'
                    saveitem['学历'] = ijob.get("degreeString")
                    saveitem['经验'] = ijob.get("workYearString")
                    saveitem["工作地区"] = ijob.get("jobAreaString")
                    print(workedu, page, saveitem)
                    redis_con.hset("joblist", ijob.get("jobId"), json.dumps(saveitem))
                except Exception as e:
                    print(f"parse job error:{e}")
            if len(job_items) <= 40:
                break

def get_sign(data, key):
    """
    @sign 解析
    """
    key = key.encode('utf-8')
    message = data.encode('utf-8')
    sign = hmac.new(key, message, digestmod=sha256).hexdigest()
    return sign


def submit_req(api,headers):
    while 1:
        try:
            res = requests.get(api,headers=headers)
            time.sleep(.1)
            return res.json()
        except Exception as e:
            print(f">>> net error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    redis_con = redis.Redis(db=3)

    #信托、证券、投资、理财、保险顾问
    dowoload_by_key(function="3334,8300,3400", workedu='', citycode="000000")