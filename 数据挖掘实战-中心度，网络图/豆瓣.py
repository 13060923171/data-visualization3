import pandas as pd
import requests
import json
from tqdm import tqdm

headers = {
    'Cookie':'bid=gBbAlXq5KgE; ll="118282"; Hm_lvt_6d4a8cfea88fa457c3127e14fb5fabc2=1700052462; Hm_lpvt_6d4a8cfea88fa457c3127e14fb5fabc2=1700052462; _gid=GA1.2.1799981941.1700052462; talionnav_show_app="0"; _ga=GA1.3.1313338170.1700052462; _gid=GA1.3.1799981941.1700052462; _vwo_uuid_v2=DD8176270AF24B5FB317C595647EF1EA1|d16d94e5434bf2932fdc5f422adf1e72; Hm_lvt_19fc7b106453f97b6a84d64302f21a04=1700052554; Hm_lpvt_19fc7b106453f97b6a84d64302f21a04=1700052554; _ga_393BJ2KFRB=GS1.2.1700052554.1.0.1700052554.0.0.0; _ga_PRH9EWN86K=GS1.2.1700052554.1.0.1700052554.0.0.0; ap_v=0,6.0; __utma=30149280.1313338170.1700052462.1700052570.1700052570.1; __utmc=30149280; __utmz=30149280.1700052570.1.1.utmcsr=m.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/; dbcl2="212088227:mVPF75oYIik"; ck=nkBv; push_doumail_num=0; __utmv=30149280.21208; push_noty_num=0; frodotk_db="6437e76f3618ecc22ca7b64da24ac646"; frodotk="40eed59980068f96c2cc80cc8013aec2"; talionusr="eyJpZCI6ICIyMTIwODgyMjciLCAibmFtZSI6ICJcdTY3MDlcdTczMmJcdTgxN2IifQ=="; _ga=GA1.2.1313338170.1700052462; __utmt=1; _ga_Y4GN1R87RG=GS1.1.1700052461.1.1.1700053261.0.0.0; __utmb=30149280.38.7.1700052626746',
    'Host':'m.douban.com',
    'Origin':'https://movie.douban.com',
    'Referer':'https://movie.douban.com/tv/',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}


def get_status(url):
    html = requests.get(url,headers=headers)
    if html.status_code == 200:
        get_data(html)
    else:
        print(html.status_code)


def get_data(html):
    content = html.json()
    items = content['items']
    list_title = []
    list_rating = []
    list_year = []
    list_racing = []
    list_director = []
    list_starring = []
    for i in items:
        title = i['title']
        list_title.append(title)
        card_subtitle = i['card_subtitle']
        rating = i['rating']['value']
        list_rating.append(rating)
        card_subtitle1 = str(card_subtitle).split('/')
        list_year.append(card_subtitle1[0])
        list_racing.append(card_subtitle1[1])
        list_starring.append(card_subtitle1[-1])

    df = pd.DataFrame()
    df['标题'] = list_title
    df['评分'] = list_rating
    df['年份'] = list_year
    df['国家'] = list_racing
    df['主演'] = list_starring
    df.to_csv('data1.csv', encoding='utf-8-sig', mode='a+', index=False, header=False)


if __name__ == '__main__':
    df = pd.DataFrame()
    df['标题'] = ['标题']
    df['评分'] = ['评分']
    df['年份'] = ['年份']
    df['国家'] = ['国家']
    df['主演'] = ['主演']
    df.to_csv('data1.csv',encoding='utf-8-sig',mode='w',index=False,header=False)

    for i in tqdm(range(0,501,20)):
        url = 'https://m.douban.com/rexxar/api/v2/tv/recommend?refresh=0&start={}&count=20&selected_categories=%7B%22%E7%B1%BB%E5%9E%8B%22:%22%22,%22%E5%BD%A2%E5%BC%8F%22:%22%E7%94%B5%E8%A7%86%E5%89%A7%22%7D&uncollect=false&sort=S&tags=%E4%B8%AD%E5%9B%BD%E5%A4%A7%E9%99%86,%E7%94%B5%E8%A7%86%E5%89%A7&ck=nkBv'.format(i)
        get_status(url)