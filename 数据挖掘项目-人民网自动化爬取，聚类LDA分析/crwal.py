from selenium import webdriver
import time
from lxml import etree
import pandas as pd
from urllib import parse


list1 = ['肺炎','病毒','新冠','口罩']

keyword = '口罩'
keywords = parse.quote(keyword)


def get_data():
    driver = webdriver.Chrome('chromedriver.exe')
    driver.get('http://liuyan.people.com.cn/messageSearch?keywords={}'.format(keywords))
    # time.sleep(5)
    # driver.find_element('xpath','//span[@class="el-checkbox__inner"]').click()
    driver.find_element('xpath','//button[@class="el-button fr el-button--danger el-button--default"]').click()
    for i in range(2100,500001,3000):
        driver.execute_script("window.scrollBy(0,{})".format(i))
        page = driver.page_source
        soup = etree.HTML(page)
        id = soup.xpath('//span[@class="t-mr1 t-ml1"]/text()')
        if len(id) != 0:
            driver.find_element('xpath', '//div[@class="mordList"]').click()
            id = soup.xpath('//span[@class="t-mr1 t-ml1"]/text()')
            save_csv(id)
            time.sleep(2)


def save_csv(data):
    df = pd.DataFrame()
    df['id'] = data
    df.to_csv('{}_all_id_data.csv'.format(keyword),encoding='utf-8-sig')


if __name__ == '__main__':
    get_data()






