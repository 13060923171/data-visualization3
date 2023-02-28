import json
import re
import time
import pandas
import redis  ##创建和数据库的交互
import requests  ## 发起网络请求
from lxml import etree ## 解析html

##定义存放过滤数据
filterdata = []

##定义请求的请求头
headers = {
    "user-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.33',
    "cookie": "urlfrom2=121122526; adfbid2=0; x-zp-client-id=c18d708b-4998-476c-90d3-074f6369dda2; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%221108527270%22%2C%22first_id%22%3A%2218639948181a62-08fc8efeaa62bf8-7d5d547c-2073600-1863994818220a%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%2C%22%24latest_utm_source%22%3A%22360PC%22%2C%22%24latest_utm_medium%22%3A%22CPC%22%2C%22%24latest_utm_campaign%22%3A%22qy%22%2C%22%24latest_utm_content%22%3A%22zh1%22%2C%22%24latest_utm_term%22%3A%22152768%22%7D%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMTg2Mzk5NDgxODFhNjItMDhmYzhlZmVhYTYyYmY4LTdkNWQ1NDdjLTIwNzM2MDAtMTg2Mzk5NDgxODIyMGEiLCIkaWRlbnRpdHlfbG9naW5faWQiOiIxMTA4NTI3MjcwIn0%3D%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%24identity_login_id%22%2C%22value%22%3A%221108527270%22%7D%2C%22%24device_id%22%3A%2218639948181a62-08fc8efeaa62bf8-7d5d547c-2073600-1863994818220a%22%7D; ssxmod_itna=Yq+h0K7IxAx0ODGxBPwQrGkt2eduCQBi=DmqKDseDcQxA5D8D6DQeGTTuNprqjrCmoP0e7QoOa0YLeIYpEeqObg4GLDmKDyFGO0eDx1q0rD74irDDxD3yD7PGmDieC0D7oZMgcTNKiOD7eDXxGCDQIsZxDaDGa8eex5E107DYW8RhPh/OYDnpGCKmKD9x0CDlPdvKr8DD5ffKxzRxfxDmb3woAGDCKDjpYC8OYDU0IznxI6Tni3=Qra5QxdLmDxt74q1CDqTmGcr+exhB6PkCD5GGxnL5xDi=4MeYD==; ssxmod_itna2=Yq+h0K7IxAx0ODGxBPwQrGkt2eduCQBi=DmqikAeotDlxWwoq03o5enjXcrn5UQru5hxx5fdhLQYOPTvtKNofliTE08fjHnESQtBfLCW1pDgGqOCO6QuRtedQvTqiL5IUW4eTt95rbPHO6gHFp1FGUeFOmpe3DQFeUD08DYFe4D=; acw_tc=276082a116766950109876258e8b1e80341a6b8aa5d4e83bd1462ac664a944; Hm_lvt_38ba284938d5eddca645bb5e02a02006=1676033971,1676034438,1676695010; zp_passport_deepknow_sessionId=acc5440bsd7e3445968bc853897c5d822d0c; selectCity_search=; Hm_lpvt_38ba284938d5eddca645bb5e02a02006=1676695258",
    "accept": 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
}

##创建数据库链接，数据库号为3
__sql = redis.Redis(db=3)


##获取需要爬取的城市code列表
def get_code_by():
    ALLCITY  = []
    ##读取json文件
    with open('city.json', 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    ##解析json文件中的城市名、城市代码、
    for item in data:
        sublist = item.get("sublist")
        for city in sublist:
            filterdata.append([city.get("name"), city.get("code")])
    ##返回解析号的城市列表数据
    return filterdata


def getcontent(baseurl):

    ##发起网络请求
    while 1:
        try:

            ##发起网络请求，指定请求地址、请求头、超时时间
            res = requests.get(
                baseurl, headers=headers, timeout=(2, 3)
            )
            time.sleep(.5)
            ##返回响应文本
            return res.text
        except Exception as e:
            print(f">>>neterror: {e}")
            ##异常请求继续重连
            time.sleep(1)


def download_code(icode, keyword):

    ##解析出爬取的城市code和名词
    code = icode[1]
    name = icode[0]

    ##构造页码
    for page in range(1, 2):
            try:
                ##构造url
                baseurl = f'https://sou.zhaopin.com/?jl={code}&kw={keyword}&p={page}'
                ##发起网络请求
                content = getcontent(baseurl)
                ele = etree.HTML(content)

                ##解析json数据，
                json_abs = re.findall(r'<script>__INITIAL_STATE__=(.*?)</script>', content, re.S)[0]
                json_dd = json.loads(json_abs)

                ##从json数据中获取岗位列表
                job_items = json_dd.get("positionList")

                ##遍历岗位列表，并写入数据库
                print(f'length: {len(job_items)}')
                for ijob in job_items:
                    saveitem = {}
                    saveitem["页面网址"] = ijob.get("positionURL")
                    saveitem['职位昵称'] = ijob.get("name")
                    saveitem['职位亮点'] = ''
                    saveitem['所在地区'] = ijob.get("cityDistrict")
                    saveitem['薪资'] = ijob.get("salary60")
                    saveitem['工作地点'] = ''
                    saveitem['城市'] = name
                    saveitem['学历'] = ijob.get("education")
                    saveitem['经验'] = ijob.get("workingExp")
                    saveitem['职位'] = ijob.get("workType")
                    saveitem['招聘人数'] = ijob.get("recruitNumber")
                    saveitem['行业'] = ijob.get("industryName")
                    saveitem['公司名称'] = ijob.get("companyName")
                    saveitem['公司地址'] = ''
                    saveitem["公司规模"] = ijob.get("companySize")
                    saveitem["发布日期"] = ijob.get("publishTime").split(" ")[0]
                    saveitem["公司主页"] = ijob.get("companyUrl")
                    saveitem["岗位职责"] = ''
                    saveitem["岗位要求"] = ''

                    print(f" 页码：{page} 数据：{saveitem}")
                    print("*" * 50 + '\n')
                    data = json.dumps(saveitem)
                    print(f"插入：{data}")
                    index_current = __sql.lpush(
                        "plzhilian_url",
                        data
                    )
                    print(name, index_current)
                if len(job_items) <= 10:
                    break
            except Exception as e:
                print(f">>>error: {e}")
                break


if __name__ == "__main__":
    # 信托、证券、投资、理财、保险顾问
    #电话销售、电话客服
    ##获取解析城市列表
    get_code_by()
    city_down = 0
    #遍历城市，下载当前城市的岗位数据
    for icode in filterdata[city_down:]:
        city_down+=1
        download_code(icode,keyword='保险顾问')
    # pandas.DataFrame([json.loads(i.decode()) for i in __sql.lrange("plzhilian_0",0,-1)]).to_excel("handle.xlsx",index=False)