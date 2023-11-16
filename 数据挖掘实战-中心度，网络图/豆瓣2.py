import requests
import pandas as pd
from lxml import etree
import chardet
import numpy as np
from urllib import parse
import time
from tqdm import tqdm

headers = {
    'Cookie':'RK=OY8wOoV4V2; ptcz=f3d3983cb2340072bd53b8931acde256fdfe2062449a339f65e321fabbfff487; video_guid=bbc5c38056794523; tvfe_search_uid=725c74c6-a7a2-4d95-9285-4a634f6cc32e; video_platform=2; pgv_pvid=3270536096; ts_uid=9383169774; qq_domain_video_guid_verify=bbc5c38056794523; _qimei_fingerprint=a48926e889d07b9237466ab1e00aa99a; _qimei_q36=; _qimei_h38=b8855faa60e49e0af7cd6a6102000001b1781b; lcad_o_minduid=opfu3bYaqjrVKGYfmgQnpCwZ4i_e0MFg; lcad_appuser=5563F7327EC26D65; player_spts=40|0|false|false|true|true|false; lcad_ad_session_id=fm5rtb6usxmig; player_defn=uhd; bucket_id=9231008; pgv_info=ssid=s6067858528; tab_experiment_str=100000#11210921; vversion_name=8.2.95; video_omgid=bbc5c38056794523; lcad_Lturn=355; lcad_LKBturn=21; lcad_LPVLturn=361; channel_tab_experiment_data=expIds=100000; ptag=; ad_play_index=17; ts_last=v.qq.com/x/search/',
    'Referer':'https://v.qq.com/',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}


def get_status(url):
    html = requests.get(url,headers=headers)
    html.encoding = chardet.detect(html.content)['encoding']
    if html.status_code == 200:
        get_data(html)
    else:
        print(html.status_code)


def get_data(html):
    content = html.text
    soup = etree.HTML(content)
    try:
        title = soup.xpath('//em[@class="hl"]/text()')[0]
    except:
        title = np.NAN
    try:
        year = soup.xpath('//span[@class="sub"]/text()')[0]
    except:
        year = np.NAN
    try:
        starring = soup.xpath('//div[@class="info_item info_item_even"]/span[@class="content"]/a/text()')
        starring1 = ' '.join(starring)
    except:
        starring1 = np.NAN

    df = pd.DataFrame()
    df['标题'] = [title]
    df['年份'] = [year]
    df['主演'] = [starring1]
    df.to_csv('data2.csv', encoding='utf-8-sig', mode='a+', index=False, header=False)

def demo():
    df = pd.read_csv('data2.csv',encoding='utf-8-sig')
    # df = df.dropna(subset=['主演'],axis=0)
    #
    # def is_Chinese(word):
    #     if '\u4e00' <= word <= '\u9fff':
    #         return word
    #     else:
    #         return np.NAN
    # df['主演'] = df['主演'].apply(is_Chinese)
    # df = df.dropna(subset=['主演'], axis=0)
    df = df.drop_duplicates()
    df.to_csv('data2.csv', encoding='utf-8-sig',index=False)


if __name__ == '__main__':
#     df = pd.DataFrame()
#     df['标题'] = ['标题']
#     df['年份'] = ['年份']
#     df['主演'] = ['主演']
#     df.to_csv('data2.csv',encoding='utf-8-sig',mode='w',index=False,header=False)
#
#     with open('获奖名单.txt', 'r', encoding='utf-8-sig') as f:
#         content = f.readlines()
#     list_content = []
#     for c in content[:-1]:
#         c1 = str(c).split('.')
#         c2 = str(c1[-1]).strip('\n').replace('》', '').replace('《', '').strip(' ')
#         list_content.append(c2)

    # with open('获奖名单2.txt', 'r', encoding='utf-8-sig') as f:
    #     content = f.readlines()
    # list_content = []
    # for c in content:
    #     c = c.strip('\n')
    #     c1 = str(c).split('/')
    #     for k in c1:
    #         k = str(k).strip(" ")
    #         list_content.append(k)
    # list_content1 = list(set(list_content))
    #
    # for l in tqdm(list_content1):
    #     keyword = parse.quote(l)
    #     url = 'https://v.qq.com/x/search/?q={}'.format(keyword)
    #     get_status(url)
    #     time.sleep(3)


    demo()