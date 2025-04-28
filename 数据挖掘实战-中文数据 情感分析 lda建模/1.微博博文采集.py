import json
import os

import pandas
import pymongo
import redis
import requests
import re
import time
from datetime import datetime
import datetime as dtime
from lxml import etree

headers = {
    "Accept": 'application/json, text/plain, */*',
    "Referer": 'https://weibo.com/u/1735618041',
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
}


def send_get(api, headers, params):
    print(f'>>>访问：{api}')
    while 1:
        try:
            res = requests.get(
                api,
                headers=headers,
                timeout=(4, 5),
                params=params,

            )
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


def get_page(word, date1,date2):
    for page in range(1, 15):
        fword = word
        burl = 'https://s.weibo.com/weibo?q={}&timescope=custom:{}:{}&Refer=g&page={}&suball=1' #&scope=ori
        url = burl.format(fword, date1, date2, page)
        while 1:
            try:
                time.sleep(.7)

                print(f">>> 正访问：{word} {date1} {page} {url}")
                r = requests.get(url, headers=headers, timeout=(4, 5))
                if '<title>Sina Visitor System</title>' in r.text or '<title>ÐÂÀËÍ¨ÐÐÖ¤</title>' in r.text:
                    print(f">>> cookie失效！")
                    time.sleep(2)
                    continue
                if r.status_code == 418:
                    print(r.status_code)
                    time.sleep(10)
                    continue
                break
            except Exception as e:
                print(f"some error:{e}")
                time.sleep(1)

        html = etree.HTML(r.content.decode('utf-8', errors='ignore'))
        divls = html.xpath('//div[@action-type="feed_list_item"]')
        pnum = len(divls)

        if '以下是您可能感兴趣的微博' in r.text:
            print(f'当前时间暂无数据：以下是您可能感兴趣的微博：{date1}')
            break

        print(f'日期：{date1}  ', f"页码：{page}", f"页码数：{pnum}")

        for div in divls:

            try:
                mid = "".join(div.xpath("./@mid"))
                link = f'https://m.weibo.cn/detail/{mid}'

                user = div.xpath('.//div[@class="info"]/div[2]/a/@href')
                img_count = len(div.xpath(".//div[@node-type='feed_list_media_prev']//img[@src]"))
                img_list = div.xpath(".//div[@node-type='feed_list_media_prev']//img[@src]/@src")
                video_count = len(div.xpath(".//div[@node-type='feed_list_media_prev']//video-player"))
                content_ = "".join(div.xpath(".//p[@node-type='feed_list_content_full']//text()")).strip()
                chaohua_list = [i for i in div.xpath(".//p[@node-type='feed_list_content_full']/a/text()") if
                                '超话' in i]
                huati_list = [i for i in div.xpath(".//p[@node-type='feed_list_content_full']/a/text()") if
                              '#' in i]
                attid = "".join(div.xpath(".//div[@class='card-act']/ul/li[3]//text()")).strip().replace("赞",
                                                                                                         "")
                coms = "".join(div.xpath(".//div[@class='card-act']/ul/li[2]//text()")).strip().replace("评论",
                                                                                                        "")
                ret = "".join(div.xpath(".//div[@class='card-act']/ul/li[1]//text()")).strip().replace("转发",
                                                                                                       "")
                if content_ == "":
                    chaohua_list = [i for i in div.xpath(".//p[@node-type='feed_list_content']/a/text()") if
                                    '超话' in i]
                    huati_list = [i for i in div.xpath(".//p[@node-type='feed_list_content']/a/text()") if '#' in i]
                    content_ = "".join(div.xpath(".//p[@node-type='feed_list_content']//text()")).strip()

                if word not in content_:
                    print(f">>> 关键字过滤：{word}")
                    continue

                if len(user) > 0:
                    userlink = user[0]
                    userid = re.findall('com\/(\d+)\?', userlink)[0]
                    username = div.xpath(
                        './/div[@class="info"]/div[2]/a/@nick-name')[0]

                    api = f'https://weibo.com/ajax/profile/info?custom={userid}'
                    response = send_get(api, headers, {}).get("data", {}).get("user")
                    user_ip_location = response.get("location")
                    if user_ip_location is None:
                        user_ip_location = response.get("ip_location")
                    user_description = response.get("description")
                    user_followers_count = response.get("followers_count")
                    user_friends_count = response.get("friends_count")
                    user_statuses_count = response.get("statuses_count")
                    user_gender = response.get("gender")
                    user_verified_reason = response.get("verified_reason")
                    user_is_svip = response.get("svip")

                else:
                    userid = ""
                    username = ""
                    user_description = ""
                    user_followers_count = ""
                    user_friends_count = ""
                    user_statuses_count = ""
                    user_gender = ""
                    user_verified_reason = ""
                    user_is_svip = ""
                    user_ip_location = ""

                time_ = div.xpath('.//div[@class="content"]/div[@class="from"]/a/text()')
                if len(time_) > 0:
                    time_ = time_[0].strip()
                else:
                    time_ = ""

                saveitem = {}
                saveitem["_id"] = mid
                saveitem["关键字"] = word
                saveitem["页面网址"] = url
                saveitem["微博发布者"] = username
                saveitem["发布者链接"] = f'https://weibo.com/u/{userid}'

                saveitem["发布时间"] = time_
                saveitem["发布内容"] = content_
                saveitem["微博链接"] = link

                saveitem["博文转发数"] = ret
                saveitem["博文评论数"] = coms
                saveitem["博文点赞数"] = attid

                saveitem["博文超话"] = chaohua_list
                saveitem["博文话题"] = huati_list
                saveitem["博文图片数"] = img_count
                saveitem["博文视频数"] = video_count

                saveitem["认证类型"] = "".join(
                    re.findall(r'<use xlink:href="#woo_svg_(.*?)">', etree.tounicode(div))[:1])
                saveitem["来自"] = "".join(div.xpath(".//div[@class='from']/a[2]/text()"))

                saveitem["用户ip属地"] = user_ip_location
                saveitem["用户标签"] = user_description
                saveitem["用户粉丝数"] = user_followers_count
                saveitem["用户关注数"] = user_friends_count
                saveitem["用户博文数"] = user_statuses_count
                saveitem["用户性别"] = user_gender
                saveitem["用户认证信息"] = user_verified_reason
                saveitem["用户vip信息"] = user_is_svip

                print(f">>> 正访问：{word} {date1}  {page}", saveitem)
                with open("./blog.txt", "a", encoding="utf-8") as f:
                    f.write(json.dumps(saveitem, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f'parse error:{e}')





def getint(t):
    try:
        d_ = "".join(re.findall('\d+', t))
        return int(d_)
    except:
        return 0

def getdaterange(starttime,endtime):
    time_str = []
    start_samp = time.mktime(time.strptime(starttime, '%Y-%m-%d'))
    while True:
        time_str.append(
            (
                time.strftime('%Y-%m-%d', time.localtime(start_samp)),
                time.strftime('%Y-%m-%d', time.localtime(start_samp+24 * 60 * 60 * 1))
            )
        )
        start_samp += 24 * 60 * 60 * 1
        if start_samp > time.mktime(time.strptime(endtime, '%Y-%m-%d')):
            break
    return time_str


def get_data(keyword,date1,date2):

    for date1,date2 in getdaterange(date1,date2):
        get_page(keyword,date1,date2)

if __name__ == '__main__':



    ##登录失效就从浏览器重新复制
    headers["cookie"] = 'SCF=Au11V-R8mBJfrVL-ElAuyRjhTOSM6PNwutXqpN2KwMUWuI9ZSQJG7aL3Q8b_OWCTKnfRWB2wIltXQC37u1erTls.; SINAGLOBAL=1553660159760.0042.1739524809891; UOR=,,tophub.today; ULV=1745080898273:78:17:1:6549893375692.823.1745080898266:1744684468325; ALF=1747810368; SUB=_2A25FAZsQDeRhGeBJ4lUY8y7EyjWIHXVmfpLYrDV8PUJbkNANLVnBkW1NRjn6U42WAlC6Jek3ORU-YLpwkZP9V2eo; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5n9zJq2EFVW1sfyACKT7rj5JpX5K-hUgL.FoqN1KM4e05ReK.2dJLoIp7LxKML1KBLBKnLxKqL1hnLBoMcS0.N1Ke71h24; XSRF-TOKEN=f5RsqnYts8BDAcoGY5SMs_Sf; PC_TOKEN=89ae7f541b; WBStorage=036f61cc|undefined; WBPSESS=S8MbxEPPgMhJABbt7NPbA2Npqq-HVmdZxa3BBs6L-oe6tsqMLeicPhMYoMkQkp3Jy1r26vqfLd4KtZXUcYcRzlNhrZl_FMhu9H0UuUteDHIjZumXO14NKOLg4f_DYVC6VQ0f2F1UoHxUFQao2COmPg=='

    for keyword in [
        '日本核废水',
        "福岛核污水",
        "核废水排海",
        "海鲜安全",
        "辐射危害",
        "IAEA报告",
        "水产品禁令"
    ]:

        get_data(keyword, '2023-08-24','2024-07-31')



    with open("./blog.txt",'r',encoding="utf-8") as f:
        all_blogs = [json.loads(i.strip()) for i in f.readlines()]

    df = pandas.DataFrame(all_blogs)
    df["博文转发数"] = df['博文转发数'].apply(getint)
    df["博文点赞数"] = df['博文点赞数'].apply(getint)
    df["博文评论数"] = df['博文评论数'].apply(getint)

    with pandas.ExcelWriter('博文表.xlsx', engine='xlsxwriter', options={'strings_to_urls': False}) as writer:
        df.to_excel(writer, index=False)
    os.remove("blog.txt")