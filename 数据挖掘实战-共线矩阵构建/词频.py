import pandas
import openpyxl
from pyecharts.charts import WordCloud
import pyecharts.options as opts

# 加载词频excel
excel = openpyxl.load_workbook("word_count.xlsx")
sheet = excel.active
# 读取数据
data = pandas.read_excel("content.xlsx")
content = data["评论"]
text = ""
for c in content:
    # 文本拼接
    text += c
word_list = text.split(" ")
dic = {}
# 词频统计
for word in word_list:
    dic[word] = dic.get(word, 0) + 1
# 词频排序
s_words = sorted(dic.items(), key=lambda x: x[1], reverse=True)
sheet.append(["词", "频"])
for s in s_words:
    sheet.append([s[0], s[1]])
# 列表推导式
data_list = [(key, value) for key, value in dic.items()]

b = (
    # series_name：添加标题 data_pair:数据 word_size_range:词的大小范围
    WordCloud().add(series_name="《青春变形记》词云图", data_pair=data_list)
        .set_global_opts(title_opts=opts.TitleOpts(
        # 标题字体大小
        title="《青春变形记》词云图", title_textstyle_opts=opts.TextStyleOpts(font_size=20),
    ), tooltip_opts=opts.TooltipOpts(is_show=True)
    )
)
b.render("词云图.html")
excel.save("word_count.xlsx")
