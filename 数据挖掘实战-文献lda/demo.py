import PyPDF4
import os


def read_pdf(name):
    # 打开您的PDF文件
    with open('./文献/{}'.format(name),'rb') as pdf_file:
        # 创建一个PDF文件阅读器对象
        reader = PyPDF4.PdfFileReader(pdf_file)
        # 获取PDF文档的页数
        num_pages = reader.numPages
        # 为所有页面初始化一个空的文本字符串
        all_text = ""
        # 循环遍历每一页，获取文本并进行拼接
        for page in range(num_pages):
            try:
                current_page = reader.getPage(page)
                page_text = current_page.extractText()
                all_text += '\n' + page_text
            except:
                pass
    return all_text


filePath = '文献'
name = os.listdir(filePath)

list_content = []
for n in name:
    print(n)
    content = read_pdf(n)
    list_content.append(content)

print(list_content)
