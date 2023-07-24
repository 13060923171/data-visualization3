import requests
from lxml import etree
import pandas as pd
from tqdm import tqdm
import time
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'cookie': 'cookie_enabled=true; cookie_enabled=true; ID=1NZNCR263R700F9; IDPWD=I95976398; COOKIE_ID=1NZNCR263R700F9; visit=1NZNCR263R700F9%7C20230724031016%7C%2F%7C%7Cend%20; user_status=A%7C; _csrf=gsYC5p_Dbo0dbVTD1ralQYskQ_hH_6is; _pxhd=anwym8h5ZYwt7JJu6N-7/kdOohpsWEnLwAuwpxjyLAIlyppbTUYGzbxc24ogwNZMdvMzBYJvY/NxDLShsf91cQ==:HrvHFNx7B4hycEwL2VYHjH6Apa41P2gZY398psJQIsUjQR2ZMO0wEqy0E9QhgYfbyGztN37PcAAXVu211tN4A2qHDy41tfZIq6Xi2do7hf4=; pxcts=3bae0f88-29c7-11ee-b31a-487068497759; _pxvid=3b2d4091-29c7-11ee-877c-503709e23f7e; NPS_bc24fa74_last_seen=1690164617547; _gid=GA1.2.783775281.1690164618; cookie_consent=allow; search=start%7Ccastello%2Bbanfi%2Bdocg%2Bbrunello%2Bdi%2Bmontalcino%2Btuscany%2Bitaly%7C1%7CHong%2BKong%7CHKD%7C%7C%7C%7C%7C%7C%7C%7C%7Ce%7Ci%7C%7C%7CN%7C%7Ca%7C%7C%7C%7CCUR%7Cend; find_tab=%23t1; adin=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE2OTAxNjQ3MDguMjQ1MzYyLCJ1aWQiOm51bGwsInVlYSI6bnVsbCwidWFjZCI6bnVsbH0.UrzN2GcKcSDC1d4SKN7TiHWe0p1Jd7kJl5NP8PEWFOU; _ga=GA1.1.1304883499.1690164618; _ga_M0W3BEYMXL=GS1.1.1690164617.1.1.1690164709.0.0.0; _px2=eyJ1IjoiNzI0ODM2YjAtMjljNy0xMWVlLTlkYWYtN2Y4NDlhMDdiZDc5IiwidiI6IjNiMmQ0MDkxLTI5YzctMTFlZS04NzdjLTUwMzcwOWUyM2Y3ZSIsInQiOjE2OTAxNjUwMTAwNDAsImgiOiI2YzJkZThhNTUxMmQ2Mzc0ZDY4MjUwYzM3OTI3MDBiZWVjOTc5Mjk4OGNhYTQ1ZjVkNmMxZGUxNTdlMGY1NDJmIn0=; _px3=c9b54889daaff77df88052cdabdb127065535e48b9f825352ffe239adfbc6f09:GIPBct01f3lQMo4RT8EynlnMHZg4W8N56nDWaBFfVd6SJTaqQQU1LGZX6NtRQDe1iLzndkqzP1+RF94BW4gE5A==:1000:QGLdkyIiBJyJq1mURoFiXyYaD3BZ5aKVqkOiWoCV3MowE2bvhzw44Un/vEqJk1S8q+NE6zq2nv10bO9baPa96+yRr7CLfWbU9zzaRpeS6eV/F7pjuJJ7YmCLjY/SlP9sC2IEYwFkFJ5rsq7JGALGrfKWmFm9DGI85qIiX0xdrhiRVTt8Xv8Q0+G0CbrxDlf9iro3ZqPcI5HXSz/bjA5RTg==; _pxde=a44a5c1b5a9236c971b9d20617ea63a865a77fcfe1037090eaf201ea11acde2e:eyJ0aW1lc3RhbXAiOjE2OTAxNjQ3MjcxMzIsImZfa2IiOjAsImlwY19pZCI6W119',
    'referer': 'https://www.wine-searcher.com/'
}


def get_status(url):
    html = requests.get(url=url,headers=headers)
    if html.status_code == 200:
        get_content(html)
    else:
        print(html.status_code)


def get_content(html):
    content = html.text
    soup = etree.HTML(content)
    href = soup.xpath('//div[@class="card card-product"]/a/@href')
    title = soup.xpath('//div[@class="card-product__name"]/a/text()')
    production = soup.xpath('//div[@class="card-product__region"]/text()')
    for h,t,p in zip(href[11:],title[11:],production[11:]):
        h1 = 'https://www.wine-searcher.com' + h
        get_reviews(h1,t,p)
        time.sleep(60)


def get_reviews(url,t,p):
    html = requests.get(url=url, headers=headers)
    if html.status_code == 200:
        content = html.text
        soup = etree.HTML(content)
        comment = soup.xpath('//div[@class="pt-2"]/text()')
        for c in tqdm(comment):
            df = pd.DataFrame()
            df['title'] = [t]
            df['production'] = [p]
            df['URL'] = [url]
            df['content'] = [c]
            df.to_csv('data1.csv', encoding='utf-8-sig', mode='a+',header=False,index=False)
    else:
        print(html.status_code)


if __name__ == '__main__':
    df = pd.DataFrame()
    df['title'] = ['title']
    df['production'] = ['production']
    df['URL'] = ['URL']
    df['content'] = ['content']
    df.to_csv('data1.csv',encoding='utf-8-sig',mode='w',header=False,index=False)
    url = 'https://www.wine-searcher.com/find/banfi/1/hong+kong'
    get_status(url)