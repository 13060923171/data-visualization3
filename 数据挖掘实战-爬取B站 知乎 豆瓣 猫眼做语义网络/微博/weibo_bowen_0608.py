# 搜索关键词、博文
import copyheaders
import pandas
import pandas as pd
import requests
import re
import json
import time
import random
from datetime import datetime
import datetime as dtime
import csv
from lxml import etree
from pathlib import Path

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3870.400 QQBrowser/10.8.4405.400'
}

# burl = 'https://s.weibo.com/weibo?q=%23{}%23&timescope=custom:{}:{}&Refer=g&page={}'
burl = 'https://s.weibo.com/weibo?q=%23{}%23&timescope=custom:{}:{}&Refer=g&page={}'
headers_m = {
    "Cookie":'SINAGLOBAL=319438869371.1958.1697274553153; SUB=_2A25IL-flDeRhGeBG7VcS9y3Nwj-IHXVr04mtrDV8PUJbkNANLWn-kW1NRhAS3AQFPsBpuQK7CiDyMa09FiqYxRbp; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhucEK.vC._Hq3xfR4Nu4mS5NHD95Qc1hqfe0M0eK.0Ws4DqcjiCrH0Uh-71Kqp; ULV=1697624743463:4:4:3:6239890865130.7705.1697624743400:1697356090320; XSRF-TOKEN=elvzsYtk_ZxA9POk_FyWHEkz; WBPSESS=rR4peIBWUq8q06lj8iZT2YLUqmERf-7UpIuaTbHm6TFBk-BfA8vVx6NjDFAAj3D0_cq6zGV6cAvunk_5eafZ4zEDxsy418lZa9TouI2sXCgk8tQ3YGmLpOOJRtIwVjh6fTpUoBMei3qlQHpO5cjdHg==',
    "Accept":'application/json, text/plain, */*',
    "Referer":'https://weibo.com/u/1735618041',
    "User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
}

def send_get(api,headers,params):
    print(f'>>>访问：{api}')
    while 1:
        try:
            res = requests.get(
                api,
                headers=headers,
                timeout=(4,5),
                params=params,

            )
            if res.status_code != 200:
                print(res.text)
                time.sleep(10)
                continue
            time.sleep(.4)
            return res.json()
        except Exception as e:
            print(f'some error:{e}')
            time.sleep(1)

def save_csv(word, row):
    result = dict(zip(keys,row))
    result["电影"] = word
    print(result)
    with open(f'comment.txt', 'a', encoding='utf-8', newline='') as f:
        f.write(json.dumps(result))
        f.write('\n')


def get_page(word, day):
    for page in range(1,51):
            try:
                url = burl.format(word, day, day, page)
                print(url)
                while 1:
                    try:
                        r = requests.get(url, headers=headers)
                        time.sleep(random.uniform(0, 0.05))
                        break
                    except Exception as e:
                        print(f'some error:{e}')
                        time.sleep(1)
                html = etree.HTML(r.text)
                divls = html.xpath('//div[@action-type="feed_list_item"]')
                pnum = len(divls)

                if pnum == 0 or '以下是您可能感兴趣的微博' in r.text:
                    print(f'当前时间暂无数据：以下是您可能感兴趣的微博：{day} -')
                    break

                print(f'日期：{day}',  f"页码：{page}", f"页码数：{pnum}")

                for div in divls:

                    mid = "".join(div.xpath("./@mid"))
                    user = div.xpath('.//div[@class="info"]/div[2]/a/@href')

                    if len(user) > 0:
                        userlink = user[0]
                        userid = re.findall('com\/(\d+)\?', userlink)[0]
                        username = div.xpath(
                            './/div[@class="info"]/div[2]/a/@nick-name')[0]

                    else:
                        username = ""
                        userid = ""
                    content_ = "".join(div.xpath(".//p[@node-type='feed_list_content_full']//text()")).strip()
                    if content_ == "":
                        content_ = "".join(div.xpath(".//p[@node-type='feed_list_content']//text()")).strip()

                    time_ = div.xpath('.//div[@class="content"]/div[@class="from"]/a/text()')
                    if len(time_) > 0:
                        time_ = time_[0].strip()
                    else:
                        time_ = ""

                    link = f'https://m.weibo.cn/detail/{mid}'
                    attid = "".join(div.xpath(".//div[@class='card-act']/ul/li[3]//text()")).strip().replace("赞", "")
                    coms = "".join(div.xpath(".//div[@class='card-act']/ul/li[2]//text()")).strip().replace("评论", "")
                    ret = "".join(div.xpath(".//div[@class='card-act']/ul/li[1]//text()")).strip().replace("转发", "")
                    row = [username, time_, content_, userid, mid, ret, coms, attid, link]
                    if str(word).lower()     in str(content_).lower():
                        save_csv(word, row)
            except Exception as e:
                print(f"some error:{e}")
                time.sleep(1)


def get_day(word, day):
    get_page(word, day)


def getdaterange(starttime, endtime):
        time_str = []
        start_samp = time.mktime(time.strptime(starttime, '%Y-%m-%d'))
        while True:
            time_str.append(
                time.strftime('%Y-%m-%d', time.localtime(start_samp))
            )
            start_samp += 24 * 60 * 60
            if start_samp > time.mktime(time.strptime(endtime, '%Y-%m-%d')):
                break
        return time_str


if __name__ == '__main__':

    words = pandas.read_excel("电影表.xlsx",dtype=str)["电影名"].to_list()
    headers[
        'Cookie'] = 'SINAGLOBAL=319438869371.1958.1697274553153; SUB=_2A25IL-flDeRhGeBG7VcS9y3Nwj-IHXVr04mtrDV8PUJbkNANLWn-kW1NRhAS3AQFPsBpuQK7CiDyMa09FiqYxRbp; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhucEK.vC._Hq3xfR4Nu4mS5NHD95Qc1hqfe0M0eK.0Ws4DqcjiCrH0Uh-71Kqp; _s_tentry=-; Apache=2362079057041.3154.1698593617393; ULV=1698593617404:6:6:1:2362079057041.3154.1698593617393:1697689962525; XSRF-TOKEN=ehvsXhIUiGQN04XJdDIdpXnj; WBPSESS=rR4peIBWUq8q06lj8iZT2SYcKVjQ3Mb993CRx61o6tgLqT7_eCIacs2q6NDpZ8ueuArD5wF3j4NO7_MRdvBMBxVrNecdVeM5JTgy68pB6wgCFB_CmAw8v2tvlzx-em1Q'
    output = 'weibo_files'
    output = Path(output).absolute()
    output.mkdir(exist_ok=True)
    keys = ['username', 'time_', 'content_', 'userid', 'mid', 'ret', 'coms', 'attid','link']
    # for word in words:
    #     save_path = output / f'博文_{word}.csv'
    #     days_sub = ['2016-01-01', '2023-11-02']
    #     days = getdaterange(days_sub[0], days_sub[1])[::-1]
    #     for iday in days:
    #         get_day(word, iday)

    with open("comment.txt",'r',encoding='utf-8') as f:
        comments = [json.loads(i.strip()) for i in f.readlines()]
    df = pandas.DataFrame(comments)

    writer = pd.ExcelWriter(r'评论下载.xlsx', engine='xlsxwriter', options={'strings_to_urls': False})
    df.to_excel(writer,index=False)
    writer.save()
