import requests
import pandas as pd
import numpy as np
from lxml import etree
import json
import time
from tqdm import tqdm
from urllib import parse


# 加载Cookies文件
def format_cookies():
    with open('./data/cookie.txt', 'r') as f:
        cookie_str = f.read()
        # cookies_list = json.load(f)
    #
    # # 将cookies转为字典
    # cookies_dict = {}
    # for cookie in cookies_list:
    #     cookies_dict[cookie['name']] = cookie['value']
    # cookie_str = '; '.join([f'{k}={v}' for k, v in cookies_dict.items()])
    return cookie_str


headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'cookie':format_cookies()
}


def get_status(url):
    html = requests.get(url,headers=headers)
    if html.status_code == 200:
        get_data(html)
    else:
        print(html.status_code)


def get_data(html):
    content = html.text
    soup = etree.HTML(content)
    href = soup.xpath('//a[@class="name"]/@href')
    URL = []
    for h in href:
        h1 = "https:" + h

        URL.append(h1)

    df = pd.DataFrame()
    df['URL'] = URL
    df.to_csv('./data/URL1.csv', encoding='utf-8-sig', mode='a+', index=False, header=False)
    time.sleep(2)


if __name__ == '__main__':
    keyword_list = ['iPhone','华为','小米','vivo','oppo','一加','iqoo','红米','荣耀','真我','三星']
    df = pd.DataFrame()
    df['URL'] = ['URL']
    df.to_csv('./data/URL1.csv', encoding='utf-8-sig', mode='w', index=False, header=False)
    for k in keyword_list:
        keyword = parse.quote(k)
        url1 = 'https://s.weibo.com/realtime?q=%23{}&rd=realtime&tw=realtime&Refer=weibo_realtime&page='.format(keyword)
        # url1 = 'https://s.weibo.com/weibo?q=%23{}&Refer=realtime_weibo&page='.format(keyword)
        for i in tqdm(range(1,51)):
            url2 = str(url1) + str(i)
            get_status(url2)
            time.sleep(2)



