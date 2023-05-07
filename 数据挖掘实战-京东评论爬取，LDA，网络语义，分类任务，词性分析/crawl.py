import requests
import pandas as pd
import time
from tqdm import tqdm
import random


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


def status_html(url,id,item):
    headers = {
        "user-agent": random.choice(user_agent),
        "x-referer-page": "https://item.jd.com/{}.html".format(id),
        "x-rp-client": "h5_1.0.0",
    }
    request = requests.get(url,headers=headers)
    if request.status_code == 200:
        get_json(request,item)
    else:
        print(request.status_code)


def get_json(html,item):
    content = html.json()
    try:
        content = content['comments']
        df = pd.DataFrame()
        for c in content:
            df['id'] = [c['id']]
            df['anonymousFlag'] = [c['anonymousFlag']]
            df['content'] = [c['content']]
            df['creationTime'] = [c['creationTime']]
            df['class'] = [item]
            df.to_csv('data.csv', encoding='utf-8-sig', mode='a+', index=False, header=False)
        time.sleep(0.2)
    except:
        pass


if __name__ == '__main__':
    df = pd.DataFrame()
    df['id'] = ['id']
    df['anonymousFlag'] = ['score']
    df['content'] = ['content']
    df['creationTime'] = ['creationTime']
    df['class'] = ['class']
    df.to_csv('data.csv',encoding='utf-8-sig',mode='w', index=False, header=False)
    url_list = ['https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683341838776&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=6839699&score=1&sortType=6&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1',
                'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683341280705&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100043945122&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1',
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683341355777&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100038004353&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342055971&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100029711927&score=0&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342130114&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=8267763&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342203437&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100019386660&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342280999&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100020059664&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342324641&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100038004397&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342372475&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100038004389&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1"]
    id_list = [6839699,100043945122,100038004353,100029711927,8267763,100019386660,100020059664,100038004397,100038004389]
    item_list = ['生鲜','家电','数码','生鲜','生鲜','家电','家电','数码','数码']
    for j, k,l in zip(url_list, id_list,item_list):
        for i in tqdm(range(0,101)):
            url = j.format(i)
            status_html(url,k,l)