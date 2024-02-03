import os.path
import time
import copyheaders
import requests
from lxml import etree

headers = copyheaders.headers_raw_to_dict(b"""
Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Connection:keep-alive
Cookie:__mta=150197897.1706616147052.1706616151677.1706616157567.3; uuid_n_v=v1; uuid=6D7BA320BF6711EE9B05F94429EFE3ED2D6CE98ED21747D689B16927A6B18AEB; _csrf=d76242a5e39bdc16dec6a95bdbba57b512d5af3dc641dc4682e5b2bab4d3c8be; Hm_lvt_703e94591e87be68cc8da0da7cbd0be2=1706616144; _lxsdk_cuid=1896db100fec8-01656acf45c0ac-7e565474-1fa400-1896db100fec8; _lxsdk=6D7BA320BF6711EE9B05F94429EFE3ED2D6CE98ED21747D689B16927A6B18AEB; Hm_lpvt_703e94591e87be68cc8da0da7cbd0be2=1706630594; __mta=150197897.1706616147052.1706616157567.1706630596318.4; _lxsdk_s=18d5b18cae9-639-6cc-a39%7C%7C20
Host:www.maoyan.com
Referer:https://www.maoyan.com/films?yearId=7&showType=3&offset=450
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0
""")

def send_get(url,headers,params):
    while 1:
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=(4,5)
            )
            time.sleep(1)
            return response
        except Exception as e:
            print(f'some error:{e}')
            time.sleep(1)


def start_year_by_yearid(yid,yname):

    for page in range(0,1000):
        offset = page * 30

        params = {
            "yearId":yid,
            "showType":3,
            "offset":offset
        }
        response = send_get('https://www.maoyan.com/films',headers,params)

        document = etree.HTML(response.text)



        movie_list = [i.split("/")[-1]   for i in document.xpath(".//div[@class='channel-detail movie-item-title']/a/@href")]
        print(yid,page,movie_list)

        with open(f"./config/{yname}.txt",'a',encoding='utf-8') as f:
            for imovid in movie_list:
                f.write(imovid)
                f.write('\n')
        if len(movie_list) <1:
            break

if  __name__ == "__main__":
    if not os.path.exists("./config"):
        os.makedirs("./config")
    # start_year_by_yearid(yid=8,yname='2013')
    # start_year_by_yearid(yid=9,yname='2014')
    # start_year_by_yearid(yid=10,yname='2015')
    # start_year_by_yearid(yid=11,yname='2016')
    # start_year_by_yearid(yid=12,yname='2017')
    start_year_by_yearid(yid=13,yname='2018')
    # start_year_by_yearid(yid=14,yname='2019')
    # start_year_by_yearid(yid=15,yname='2020')
    # start_year_by_yearid(yid=16,yname='2021')
    # start_year_by_yearid(yid=17,yname='2022')
    # start_year_by_yearid(yid=18,yname='2023')
    # start_year_by_yearid(yid=19,yname='2024')
