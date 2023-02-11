from selenium import webdriver
from selenium.webdriver import Chrome, ChromeOptions
import time
from lxml import etree
import pandas as pd
from tqdm import tqdm
import random
import os


#等把全部的ID获取好之后，直接运行这个代码就好了，这个代码是自动跑的，没有第一个那么麻烦
def get_tid():
    filePath = './数据/'
    keyword = ['肺炎','病毒','新冠','发烧','疫情','核酸']
    for i in os.listdir(filePath):
        for k in tqdm(keyword):
            if k in i:
                df = pd.read_csv('./数据/{}'.format(i))
                df = df.drop_duplicates(subset='id', keep='first')
                id = list(df['id'])
                for i in tqdm(id):
                    tid = str(i).split(":")
                    tid = tid[1]
                    get_data(tid,k)


def get_data(tid,keyword):
    option = ChromeOptions()
    option.headless = True
    driver = webdriver.Chrome('chromedriver.exe',options=option)
    driver.get('http://liuyan.people.com.cn/threads/content?tid={}&from=search'.format(tid))
    # driver.get('http://liuyan.people.com.cn/threads/content?tid=12544633&from=search')
    page = driver.page_source
    soup = etree.HTML(page)
    time.sleep(0.1)
    try:
        title = soup.xpath('//h1[@class="fl"]/text()')[0]
    except:
        title = ''
    try:
        date_time = soup.xpath('//li[@class="replyMsg"]/span[2]/text()')[0]
    except:
        date_time = ''
    try:
        ask_questions = soup.xpath('//p[@id="replyContentMain"]/text()')[0]
    except:
        ask_questions = ''
    try:
        Message_object = soup.xpath('//div[@class="replyObject fl"]/text()')[0]
    except:
        Message_object = ''
    try:
        reply = soup.xpath('//p[@class="handleContent noWrap sitText"]/text()')[0]
    except:
        reply = ''

    try:
        reply_time = soup.xpath('//div[@class="handleTime"]/text()')[0]
    except:
        reply_time = ''

    try:
        #投诉/求助
        domain_name = soup.xpath('//p[@class="typeNameD"]/text()')[0]
    except:
        domain_name=''
    try:
        ask_type = soup.xpath('//p[@class="domainName"]/text()')[0]
    except:
        ask_type = ''


    data = pd.DataFrame()
    data['关键词'] = [keyword]
    data['标题'] = [title]
    data['提问时间'] = [date_time]
    data['提问内容'] = [ask_questions]
    data['地方'] = [Message_object]
    data['答复时间'] = [reply_time]
    data['官方答复'] = [reply]
    data['问题领域'] = [domain_name]
    data['提问类型'] = [ask_type]
    data.to_csv('data.csv', encoding='utf-8-sig', mode='a+', header=False, index=False)
    # 关闭当前窗口页面
    driver.close()


if __name__ == '__main__':
    data = pd.DataFrame()
    data['关键词'] = ['关键词']
    data['标题'] = ['标题']
    data['提问时间'] = ['提问时间']
    data['提问内容'] = ['提问内容']
    data['地方'] = ['地方']
    data['答复时间'] = ['答复时间']
    data['官方答复'] = ['官方答复']
    data['问题领域'] = ['问题领域']
    data['提问类型'] = ['提问类型']
    data.to_csv('data.csv',encoding='utf-8-sig',mode='w',header=False,index=False)
    get_tid()







