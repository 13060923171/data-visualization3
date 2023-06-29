import requests
import random
import pandas as pd
from tqdm import tqdm
from lxml import etree
import time
import numpy as np

def main_ur(i):
    user_agent = [
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
        "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
    ]
    headers = {
        'user-agent': random.choice(user_agent),
        'Cookie':'输入属于你的Cookie值'
    }

    html = requests.get(i,headers=headers)
    content = html.text
    soup = etree.HTML(content)
    node = soup.xpath('//div[@class="card-feed"]')
    act = soup.xpath('//div[@class="card-act"]/ul')
    for n,a in zip(node,act):
        try:
            name = n.xpath('./div[@node-type="like"]/div/div[2]/a/@nick-name')[0]

        except:
            name = np.NaN
        # try:
        #     title = n.xpath('./div[@class="avator"]/a/span/@title')[0]
        #
        # except:
        #     title = np.NaN
        try:
            timedate = n.xpath('./div[@node-type="like"]/div[2]/a/text()')[0]
        except:
            timedate = np.NaN
        try:
            comtent = n.xpath('./div[@node-type="like"]/p[@node-type="feed_list_content"]/text()')
            comtent1 = ' '.join(comtent)
        except:
            comtent1 = np.NaN
        try:
            dianzan = a.xpath('./li[3]/a/button/span[2]/text()')[0]

        except:
            dianzan =  np.NaN
        try:
            zhuanfa = a.xpath('./li[1]/a/text()')[0]

        except:
            zhuanfa =  np.NaN
        try:
            pinglun = a.xpath('./li[2]/a/text()')[0]

        except:
            pinglun =  np.NaN
        df = pd.DataFrame()
        df['时间'] = [timedate]
        df['博主'] = [name]
        df['内容'] = [comtent1]
        df['点赞'] = [dianzan]
        df['转发'] = [zhuanfa]
        df['评论'] = [pinglun]
        df.to_csv('钟薛高.csv', index=None, header=None, mode='a+', encoding='utf-8-sig')
    time.sleep(0.2)


if __name__ == '__main__':
    df = pd.DataFrame()
    df['时间'] = ['时间']
    df['博主'] = ['博主']
    df['内容'] = ['内容']
    df['点赞'] = ['点赞']
    df['转发'] = ['转发']
    df['评论'] = ['评论']
    df.to_csv('钟薛高.csv',index=None,header=None,mode='w',encoding='utf-8-sig')
    rng = pd.date_range(start='5/31/2022', end='7/31/2022')
    list_time = []
    for r in rng:
        r = str(r).split(" ")
        list_time.append(r[0])
    for l in tqdm(range(len(list_time)-1)):
        for i in range(1,51,1):
            url = 'https://s.weibo.com/weibo?q=%E9%92%9F%E8%96%9B%E9%AB%98&typeall=1&suball=1&timescope=custom:{}:{}&Refer=g&page={}'.format(list_time[l],list_time[l+1],i)
            main_ur(url)

