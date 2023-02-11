from selenium import webdriver
from selenium.webdriver.support.select import Select
import time
from lxml import etree
import pandas as pd
from urllib import parse
from datetime import datetime

list1 = ['肺炎','病毒','新冠','发烧','疫情','核酸']

keyword = '核酸'
keywords = parse.quote(keyword)


def get_data():
    a = datetime.now().timestamp()
    driver = webdriver.Chrome('chromedriver.exe')
    driver.get('http://liuyan.people.com.cn/messageSearch?keywords={}'.format(keywords))
    time.sleep(9)
    # driver.find_element('xpath','//span[@class="el-checkbox__inner"]').click()
    driver.find_element('xpath','//button[@class="el-button fr el-button--danger el-button--default"]').click()
    for i in range(0,300001,5000):
        driver.execute_script("window.scrollBy(0,{})".format(i))
        page = driver.page_source
        soup = etree.HTML(page)
        id = soup.xpath('//span[@class="t-mr1 t-ml1"]/text()')
        time.sleep(0.2)
        if len(id) != 0:
            try:
                driver.find_element('xpath', '//div[@class="mordList"]').click()
                id = soup.xpath('//span[@class="t-mr1 t-ml1"]/text()')
                save_csv(id, a)
            except:
                driver.execute_script("window.scrollBy(0,{})".format(i))
                page = driver.page_source
                soup = etree.HTML(page)
                id = soup.xpath('//span[@class="t-mr1 t-ml1"]/text()')
                save_csv(id, a)
                driver.close()


def save_csv(data,a):
    df = pd.DataFrame()
    df['id'] = data
    df.to_csv('./数据/{}_{}_id_data.csv'.format(int(a),keyword),encoding='utf-8-sig')


if __name__ == '__main__':
    get_data()

#这里是控制页面的停留时间，如果到时候不好选择省份，就加延迟就好了，这里1指的是1秒，5代表5秒




