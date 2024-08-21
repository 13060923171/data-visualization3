import urllib.parse
import pandas
import pymongo
import requests
import re
import time
from datetime import datetime
import datetime as dtime
from lxml import etree

headers = {
    "Cookie": 'SINAGLOBAL=1872946714867.0852.1717913645879; SCF=AqKWdiE3VObm6S9YvufIOMJI8IXWkvrIULdPctXrMzEmqqr1Ljun1QeBGeRgq3hlJaxAf09LuFQaAYy-TEVVskE.; UOR=,,cn.bing.com; ALF=1726299650; SUB=_2A25LucFSDeRhGeBJ4lUY8y7EyjWIHXVot1yarDV8PUJbkNANLXLHkW1NRjn6U5fX8VNag2kvflCgNCfOrBzMSMgI; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5n9zJq2EFVW1sfyACKT7rj5JpX5KMhUgL.FoqN1KM4e05ReK.2dJLoIp7LxKML1KBLBKnLxKqL1hnLBoMcS0.N1Ke71h24; _s_tentry=weibo.com; Apache=2398116145061.8794.1723733844091; ULV=1723733844143:46:12:7:2398116145061.8794.1723733844091:1723707654885',
    "Accept": 'application/json, text/plain, */*',
    "Referer": 'https://weibo.com/u/1735618041',
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
}

def send_get_html(api, headers, params):
    print(f'>>>访问：{api}')
    while 1:
        try:
            time.sleep(.4)
            res = requests.get(
                api,
                headers=headers,
                timeout=(4, 5),
                params=params,

            )
            if '访问频次过高' in res.text:
                print('访问频次过高')
                time.sleep(10)
                continue
            if res.status_code == 418:
                print(res.status_code)
                time.sleep(1)
                continue
            if res.status_code == 400:
                print(res.text)
                time.sleep(1)
                return {}
            if res.status_code != 200:
                print(res.text)
                time.sleep(10)
                continue
            return res
        except Exception as e:
            print(f'some error:{e}')
            time.sleep(1)



def send_get(api, headers, params):
    print(f'>>>访问：{api}')
    while 1:
        try:
            time.sleep(.4)
            res = requests.get(
                api,
                headers=headers,
                timeout=(4, 5),
                params=params,

            )
            if '访问频次过高' in res.text:
                print('访问频次过高')
                time.sleep(10)
                continue
            if res.status_code == 418:
                print(res.status_code)
                time.sleep(1)
                continue
            if res.status_code == 400:
                print(res.text)
                time.sleep(1)
                return {}
            if res.status_code != 200:
                print(res.text)
                time.sleep(10)
                continue
            return res.json()
        except Exception as e:
            print(f'some error:{e}')
            time.sleep(1)




def get_page(word, day):
        for page in range(1, 51):
            try:
                burl = 'https://s.weibo.com/weibo?q={}&timescope=custom:{}:{}&Refer=g&page={}'
                url = burl.format(word, day, day, page)

                r = send_get_html(url,headers,{})

                html = etree.HTML(r.text)
                divls = html.xpath('//div[@action-type="feed_list_item"]')
                pnum = len(divls)

                if '以下是您可能感兴趣的微博' in r.text:
                    print(f'当前时间暂无数据：以下是您可能感兴趣的微博：{day}')
                    break

                print(f'日期：{day} ', f"页码：{page}", f"页码数：{pnum}")

                for div in divls:

                    try:



                        mid = "".join(div.xpath("./@mid"))
                        link = f'https://m.weibo.cn/detail/{mid}'

                        if "".join(div.xpath(".//div[@class='card-comment']/@style")) != "":
                            print(f"当前帖子属于转发贴：{link}")
                            flag = '转发贴'
                        else:
                            flag = '原贴'




                        user = div.xpath('.//div[@class="info"]/div[2]/a/@href')
                        img_count = len(div.xpath(".//div[@node-type='feed_list_media_prev']//img[@src]"))
                        video_count = len(div.xpath(".//div[@node-type='feed_list_media_prev']//video-player"))
                        content_ = "".join(div.xpath(".//p[@node-type='feed_list_content_full']//text()")).strip()
                        chaohua_list = [i for i in div.xpath(".//p[@node-type='feed_list_content_full']/a/text()") if '超华' in i]
                        huati_list = [i for i in div.xpath(".//p[@node-type='feed_list_content_full']/a/text()") if '#' in i]
                        attid = "".join(div.xpath(".//div[@class='card-act']/ul/li[3]//text()")).strip().replace("赞",
                                                                                                                 "")
                        coms = "".join(div.xpath(".//div[@class='card-act']/ul/li[2]//text()")).strip().replace("评论",
                                                                                                                "")
                        ret = "".join(div.xpath(".//div[@class='card-act']/ul/li[1]//text()")).strip().replace("转发",
                                                                                                               "")


                        if content_ == "":
                            chaohua_list = [i for i in div.xpath(".//p[@node-type='feed_list_content']/a/text()") if '超华' in i]
                            huati_list = [i for i in div.xpath(".//p[@node-type='feed_list_content']/a/text()") if'#' in i]
                            content_ = "".join(div.xpath(".//p[@node-type='feed_list_content']//text()")).strip()
                        if len(user) > 0:
                            userlink = user[0]
                            userid = re.findall('com\/(\d+)\?', userlink)[0]
                            username = div.xpath('.//div[@class="info"]/div[2]/a/@nick-name')[0]


                        else:
                            userid = ""
                            username = ""

                            pass

                        time_ = div.xpath('.//div[@class="content"]/div[@class="from"]/a/text()')
                        if len(time_) > 0:
                            time_ = time_[0].strip()
                        else:
                            time_ = ""




                        saveitem = {}
                        saveitem["_id"] = mid
                        saveitem["搜索词条"] = word
                        saveitem["博文发布时间"] = time_
                        saveitem["博文内容"] = content_
                        saveitem["博文转发数"] = ret
                        saveitem["博文评论数"] = coms
                        saveitem["博文点赞数"] = attid
                        saveitem["博文链接"] = link
                        saveitem["博文超华"] = chaohua_list
                        saveitem["博文话题"] = huati_list
                        saveitem["博文图片数"] = img_count
                        saveitem["博文视频数"] = video_count
                        saveitem["博文类型"] = flag

                        saveitem["用户id"] = userid
                        saveitem["用户名"] = username


                        print(f">>> 正访问：{word} {day}  {page}", saveitem)



                        try:
                                blogs_searchs.append(saveitem)
                        except Exception as e:
                                print(f'save error:{e}')


                    except Exception as e:
                        print(f'parse error:{e}')
            except Exception as e:
                print(f'parse rquests error:{e}')



def get_day(word, day):
    get_page(word, day)

def pair_list(numbers):
    pairs = []
    for i in range(0, len(numbers), 2):
        if i+1 < len(numbers):
            pairs.append([numbers[i], numbers[i+1]])
    return pairs


def start_craw(word,start,end):
    all_tempday = []
    days = [start, end]

    tempday = days[0]
    endday = datetime.strptime(days[-1], '%Y-%m-%d')

    while 1:
        print('---------------------------')
        print('now:', datetime.now())

        all_tempday.append(tempday)
        t = datetime.strptime(tempday, '%Y-%m-%d')
        next_day = t + dtime.timedelta(days=1)
        if next_day > endday:
            break
        tempday = next_day.strftime('%Y-%m-%d')

    print(all_tempday)
    for iday in all_tempday[::-1]:
        get_day(word, iday)

if __name__ == '__main__':

    blogs_searchs = []
    start_craw(word="董宇辉", start='2022-06-01', end='2024-07-31')

    df = pandas.DataFrame(blogs_searchs)
    writer = pandas.ExcelWriter(f'./data/博文表.xlsx', engine='xlsxwriter', options={'strings_to_urls': False})
    df.to_excel(writer, index=False)
    writer.close()














