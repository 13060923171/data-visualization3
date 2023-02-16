import requests
import random
import pandas as pd
from tqdm import tqdm
from lxml import etree
import time
import numpy as np
from urllib import parse
#忽视告警信息
import warnings
warnings.filterwarnings("ignore")


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
    # 构建请求头
    headers = {
        #请求网站
        "referer": "https://s.weibo.com/weibo?q={}&typeall=1&suball=1&timescope=custom%3A{}%3A{}&Refer=g".format(keyword,list_time[l],list_time[l+1]),
        #请求头
        'user-agent': random.choice(user_agent),
        #请求cookie，这里到时候最好改为自己的，因为有时效性
        "cookie": "SINAGLOBAL=1903674588878.3147.1642225343191; UOR=www.google.com,weibo.com,login.sina.com.cn; XSRF-TOKEN=jjxj_FvCR3eJZYfxpCmRqy34; login_sid_t=d8402ffbb050630d2df913d4c9123ed8; cross_origin_proto=SSL; _s_tentry=weibo.com; Apache=2196685145962.88.1676119341893; ULV=1676119341897:11:1:1:2196685145962.88.1676119341893:1666964225108; SSOLoginState=1676119362; SCF=Au12ZUk5ZoqVeZ0uiYs6XR3pCJ4Ej4CU6-4dgqmIxj7pKuV3EDnU4eC9VtwBwQf89gZwmOy16erKfwFbNhJNmzA.; SUB=_2A25O4_0SDeRhGeNN6lsS-CrJyz-IHXVtmWnarDV8PUNbmtANLUrDkW9NSdnQVEDeTbr9Pg8KZxtNX1qYumJwBjmH; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5GwlB4mf13pUNVQ0MzU9ZV5JpX5KzhUgL.Fo-0eK.01hBfehe2dJLoIEBLxKqL1-eL1h.LxKML12eLB-zLxKnL1h-LB.zLxK-LBKqL1Kqt; ALF=1707655360; WBPSESS=m_jIqsQWDgyNVMOUb5UStptZjOxn97iXGZXHt6-H43RlScxOkkpcYteK6DtItZSliem3vaxrVyAUDTq61dtrQ9M9KF37z_VdtCcCS5AMNS2NrB8YfLj5bjmp7kinMslxw60fjoyyFo5jDmVtqNTUHg==",
    }
    #获取网页源码
    html = requests.get(i,headers=headers,verify=False)
    #把源码格式改为str格式
    content = html.text
    #使用xpath语法来定位元素
    soup = etree.HTML(content)
    #定位内容框架
    node = soup.xpath('//div[@class="card-feed"]')
    #定位点赞框架
    act = soup.xpath('//div[@class="card-act"]/ul')
    for n,a in zip(node,act):
        try:
            #获取博主的名字
            name = n.xpath('./div[@node-type="like"]/div/div[2]/a/@nick-name')[0]
        except:
            name = " "
        # try:
        #     title = n.xpath('./div[@class="avator"]/a/span/@title')
        # except:
        #     title = ' '
        try:
            #获取发帖的时间
            timedate = n.xpath('./div[@node-type="like"]/div[2]/a[1]/text()')[0]
        except:
            timedate = " "
        try:
            #获取正文内容
            comtent = n.xpath('./div[@node-type="like"]/p[@node-type="feed_list_content"]/text()')
            comtent1 = ' '.join(comtent)
        except:
            comtent1 = " "
        try:
            #获取点赞数量
            dianzan = a.xpath('./li[3]/a/button/span[2]/text()')[0]
        except:
            dianzan = " "
        try:
            #获取转发数量
            zhuanfa = a.xpath('./li[1]/a/text()')[1]
        except:
            zhuanfa = " "
        try:
            #获取评论数据
            pinglun = a.xpath('./li[2]/a/text()')[0]

        except:
            pinglun = " "
        #把上面获取到的内容保存为csv文件
        df = pd.DataFrame()
        df['时间'] = [timedate]
        df['博主'] = [name]
        df['内容'] = [comtent1]
        df['点赞'] = [dianzan]
        df['转发'] = [zhuanfa]
        df['评论'] = [pinglun]
        df.to_csv('{}.csv'.format(k), index=None, header=None, mode='a+', encoding='utf-8-sig')
    #设置停止时间，1毫秒，防止请求过快，导致被封
    time.sleep(0.1)


if __name__ == '__main__':
    #设置关键词
    keywords = ['股票','科比','詹姆斯历史得分王','#ChatGPT#']
    #读取关键词
    for k in keywords:
        #把关键词转化编码格式
        keyword = parse.quote(k)
        #创立csv文件
        df = pd.DataFrame()
        df['时间'] = ['时间']
        df['博主'] = ['博主']
        df['内容'] = ['内容']
        df['点赞'] = ['点赞']
        df['转发'] = ['转发']
        df['评论'] = ['评论']
        df.to_csv('{}.csv'.format(k),index=None,header=None,mode='w',encoding='utf-8-sig')
        #设置时间序列
        rng = pd.date_range(start='1/10/2023', end='2/10/2023')
        list_time = []
        #读取时间
        for r in rng:
            r = str(r).split(" ")
            list_time.append(r[0])
        #把时间按照天的维度，去读取
        for l in tqdm(range(len(list_time)-1)):
            #每一天获取的数据为50页
            for i in range(1,51,1):
                #构建对应的URL
                url = 'https://s.weibo.com/weibo?q=%23{}%23&typeall=1&suball=1&suball=1&timescope=custom%3A{}%3A{}&Refer=g&page={}'.format(keyword,list_time[l],list_time[l+1],i)
                #开始运行函数
                main_ur(url)
