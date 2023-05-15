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
        'cookie': 'shshshfpa=e604c3d9-8610-35f1-22ef-5c0c6e835a3b-1679034041; shshshfpx=e604c3d9-8610-35f1-22ef-5c0c6e835a3b-1679034041; shshshfpb=eHKyz1OW0zi1qoYhKzxKKUg; __jdu=16790340420591442817007; shshshfp=13aa5062aa1b7ea1879b1515213f1fb0; __jdv=76161171|direct|-|none|-|1683278923672; areaId=19; PCSYCityID=CN_440000_440300_0; ipLoc-djd=19-1607-4773-62121; joyya=1683358902.0.17.1p4x8mo; mba_muid=16790340420591442817007; jcap_dvzw_fp=HtmyChSDwatWkpaCQxE4as8ou3S_8GRQkbbncLuM2CReDwxfptOHXlja8oCFCBrtPR281zwD3Rce8PskBum4sQ==; x-rp-evtoken=N-nAb5Oj6OS1u8hkvixIgHmeflIx8-eMF0b5e2nMKV3D7JUiYiR4b7krOVORBgiHMLj7OebiR2O63n_x93oHGYzQfS8f4KYEaYewpcoxfPX45g1t8-0VObl4DTLPcHcqqH_Ljghc7j_7A77LxePB_5fisPIiv-tlf7MwyUC8BsIwggLWPVL4333jvm4AiqZRk-EqHKRn5wb4g0uk0l5qaY41DkZBplzKd5ED3PBfn_g%3D; unpl=JF8EAKBnNSttWhlcVxkGEkEUTlVVWwoKG0RXbDQABFpfHldSHlJOFxR7XlVdXxRKEx9vZhRUXVNPVg4YBSsSEHtdVV9eDkIQAmthNWRVUCVUSBtsGHwQBhAZbl4IexcCX2cCUlVZSVEEHgEaFhhLWFFdVAhNFgJpVwRkXV57ZDUaMhoiEXsWOl8QCEwRC25lAFVYW0pQDRsHHhEZS1tVX1s4SicA; jsavif=1; __jda=122270672.16790340420591442817007.1679034042.1683517298.1683690811.9; __jdc=122270672; 3AB9D23F7A4B3C9B=A2XJGV2ZYWA4MOCKI56Z5NBXQUT3T4YGXHOTB3R6GHDWNBFETIGHNVT2C2IAFB6TSTWT5BZGTR2VQCAB6JCWGKVD7M; token=a96e06549b5fd7a65bd93deceacafd00,2,935384; RT="z=1&dm=jd.com&si=wjzietd6hof&ss=lhh61lov&sl=1&tt=ay3&nu=a5103d71b53726e2588514a2c2c02bfc&cl=lbda&ld=qk3o&r=a5103d71b53726e2588514a2c2c02bfc&ul=qk3p&hd=qkc1"; __tk=AujF4VnF4DXqAUs0Azn1AVsz4DWykrnDkVoxADW1jDg,2,935384; 3AB9D23F7A4B3CSS=jdd03A2XJGV2ZYWA4MOCKI56Z5NBXQUT3T4YGXHOTB3R6GHDWNBFETIGHNVT2C2IAFB6TSTWT5BZGTR2VQCAB6JCWGKVD7MAAAAMIAPRGRCYAAAAAC6UF4W6XIIZGAEX; _gia_d=1; __jdb=122270672.12.16790340420591442817007|9.1683690811',

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
            df.to_csv('data1.csv', encoding='utf-8-sig', mode='a+', index=False, header=False)
        time.sleep(0.5)
    except:
        pass


if __name__ == '__main__':
    df = pd.DataFrame()
    df['id'] = ['id']
    df['anonymousFlag'] = ['score']
    df['content'] = ['content']
    df['creationTime'] = ['creationTime']
    df['class'] = ['class']
    df.to_csv('data1.csv',encoding='utf-8-sig',mode='w', index=False, header=False)
    url_list = ['https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683341838776&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=6839699&score=1&sortType=6&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1',
                'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683341280705&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100043945122&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1',
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683341355777&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100038004353&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342055971&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100029711927&score=0&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342130114&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=8267763&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342203437&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100019386660&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342280999&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100020059664&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342324641&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100038004397&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1",
                "https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683342372475&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683278924.1683339382.6&productId=100038004389&score=1&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1"]

    url_list1 = ['https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683692096460&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683517298.1683690811.9&productId=6839699&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1',
                 'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683692138811&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683517298.1683690811.9&productId=100043945122&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
                 'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683692174038&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683517298.1683690811.9&productId=100038004353&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
                 'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683692210910&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683517298.1683690811.9&productId=100029711927&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
                 'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683692247847&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683517298.1683690811.9&productId=8267763&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
                 'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683692278539&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683517298.1683690811.9&productId=100019386660&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
                 'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683692324856&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683517298.1683690811.9&productId=100020059664&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
                 'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683692356369&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683517298.1683690811.9&productId=100038004397&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1',
                 'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1683692388025&loginType=3&uuid=122270672.16790340420591442817007.1679034042.1683517298.1683690811.9&productId=100038004389&score=3&sortType=5&page={}&pageSize=10&isShadowSku=0&fold=1']
    id_list = [6839699,100043945122,100038004353,100029711927,8267763,100019386660,100020059664,100038004397,100038004389]
    item_list = ['生鲜','家电','数码','生鲜','生鲜','家电','家电','数码','数码']
    for j, k,l in zip(url_list1, id_list,item_list):
        for i in tqdm(range(0,51)):
            url = j.format(i)
            status_html(url,k,l)