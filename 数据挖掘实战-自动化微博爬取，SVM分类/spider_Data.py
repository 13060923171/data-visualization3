import datetime
import sys
import time
import json
import numpy as np
import pandas as pd
from lxml import etree
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options


def login():
    driver.get("https://s.weibo.com/weibo?q=%E7%96%AB%E6%83%85&Refer=realtime_weibo&page=1")
    input("请手动登录，登录后请回车")
    time.sleep(2)
    print("登录成功!")

#
# def save_cookie():
#     # 保存cookie到本地文件
#     with open('./data/cookie.txt', 'w') as file:
#         file.write(json.dumps(driver.get_cookies()))


def get_data():
    driver.get(url)
    if url == f"https://weibo.com/hot/weibo/102803":
        input("搜索完毕请回车")
        time.sleep(1.5)

    # # Get scroll height
    # last_height = driver.execute_script("return document.body.scrollHeight")
    # last_href = None
    # while True:
    #     # Scroll down to bottom
    #     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    #
    #     # Wait to load page
    #     time.sleep(2)
    #
    #     # Calculate new scroll height and compare with last scroll height
    #     new_height = driver.execute_script("return document.body.scrollHeight")
    #     page = driver.page_source
    #     soup = etree.HTML(page)
    #     href = soup.xpath('//div[@class="woo-box-flex woo-box-alignCenter head_nick_1yix2"]/a/@href')
    #     # Check if href has changed
    #     if href != last_href:
    #         last_href = href
    #         print(href)
    #         get_user(href)
    #     else:
    #         # If href has not changed, wait for a moment and try again
    #         wait = WebDriverWait(driver, 10)
    #         wait.until(EC.presence_of_element_located(
    #             (By.XPATH, '//div[@class="woo-box-flex woo-box-alignCenter head_nick_1yix2"]/a')))
    #         continue
    #     if new_height == last_height:
    #         break
    #     last_height = new_height

    # driver.close()


def get_user(href):
    driver.get(url)
    if url == f"https://weibo.com/hot/weibo/102803":
        input("搜索完毕请回车")
        time.sleep(1.5)
    for h in tqdm(href):
        driver.get(h)
        time.sleep(1.5)
        # 定位元素
        try:
            # element = driver.find_element(By.CSS_SELECTOR,'.woo-box-flex.woo-box-alignCenter.woo-box-justifyCenter.ProfileHeader_opt_1fOBM.ProfileHeader_capsuleOpt_2xcxh')
            # # 模拟点击这个元素
            # element.click()
            page = driver.page_source
            soup = etree.HTML(page)
            name = soup.xpath('//div[@class="ProfileHeader_name_1KbBs"]/text()')[0].strip(' ')
            region = soup.xpath('//div[@class="woo-box-flex woo-box-alignStart"]/div[1][@class="woo-box-item-flex ProfileHeader_con3_Bg19p"]/text()')[0].strip(' ')
            introduction = soup.xpath('//div[@class="woo-box-flex woo-box-alignStart"]/div[2][@class="woo-box-item-flex ProfileHeader_con3_Bg19p"]/text()')[0].strip(' ')
            fan = soup.xpath('//div[@class="woo-box-flex woo-box-alignCenter ProfileHeader_h4_gcwJi"]/a/span/span/text()')[0]
            focus_on = soup.xpath('//div[@class="woo-box-flex woo-box-alignCenter ProfileHeader_h4_gcwJi"]/a/span/span/text()')[1]
            number = soup.xpath('//div[@class="wbpro-screen-v2 woo-box-flex woo-box-alignCenter woo-box-justifyBetween"]/div/text()')[0].strip(' ')
            df = pd.DataFrame()
            df['URL'] = [h]
            df['昵称'] = [name]
            df['简介'] = [introduction]
            df['ip属地'] = [region]
            df['粉丝量'] = [fan]
            df['关注量'] = [focus_on]
            df['博文数'] = [number]
            df.to_csv('./data/data.csv', encoding='utf-8-sig', index=False, header=False, mode='a+')
            time.sleep(1)
        except:
            pass
    # driver.back()


if __name__ == '__main__':
    # df = pd.DataFrame()
    # df['URL'] = ['URL']
    # df['昵称'] = ['昵称']
    # df['简介'] = ['简介']
    # df['ip属地'] = ['ip属地']
    # df['粉丝量'] = ['粉丝量']
    # df['关注量'] = ['关注量']
    # df['博文数'] = ['博文数']

    # df.to_csv('data.csv',encoding='utf-8-sig',index=False,header=False,mode='w')
    data = pd.read_csv("./data/URL1.csv")
    data = data.drop_duplicates(subset='URL',keep='first')
    href = data['URL'].tolist()
    # options = Options()
    # options.add_argument('--disable-gpu')
    driver = webdriver.Chrome('chromedriver.exe')
    # driver = webdriver.Edge(options=options,executable_path='msedgedriver.exe')
    login()
    # save_cookie()
    url = 'https://weibo.com/hot/weibo/102803'
    get_user(href)
    sys.exit()