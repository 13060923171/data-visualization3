import time
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from lxml import etree
import openpyxl


class IMDB:
    def __init__(self):
        options = Options()
        options.add_argument("--headless")
        # 初始化 chrome 爬取
        self.chrome = webdriver.Chrome(executable_path="./chromedriver", options=options)
        # 打开excel
        self.excel = openpyxl.load_workbook("content.xlsx")
        # 选择当前sheet
        self.sheet = self.excel.active
        # 添加表头
        self.sheet.append(["评论"])

    def get_data(self):
        self.chrome.get("https://www.imdb.com/title/tt8097030/reviews?ref_=tt_ov_rt")
        time.sleep(3)
        for page in range(17):
            # 翻页爬取
            self.chrome.execute_script("document.documentElement.scrollTop=100000")
            self.chrome.find_element_by_id("load-more-trigger").click()
            time.sleep(2)
            print("正在爬取第{}页".format(page))
        html = etree.HTML(self.chrome.page_source)
        # 解析评论内容
        contents = html.xpath("//div[@class='lister-item-content']/div[@class='content']/div[1]/text()")
        for content in contents:
            # 将评论添加到excel
            self.sheet.append([content])
        # 保存爬取结果
        self.excel.save("content.xlsx")
        print("爬取数据结束")
        self.chrome.quit()


if __name__ == '__main__':
    IMDB().get_data()
