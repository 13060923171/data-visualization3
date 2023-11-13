import json

import pandas
import requests
import time
import re


def intToStrTime(a):
    b = time.localtime(a)  # 转为日期字符串
    c = time.strftime("%Y/%m/%d %H:%M:%S", b)  # 格式化字符串
    return c


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3970.5 Safari/537.36',
    'Referer': 'https://www.bilibili.com/'
}


def validateTitle(title):
    re_str = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(re_str, "_", title)  # 替换为下划线
    return new_title



def send_get(url,headers,params):
    while 1:
        try:
            print(f"访问：{url}")
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=(4,5)
            )
            time.sleep(.8)
            return response.json()
        except Exception as e:
            print(f"some error:{e}")
            time.sleep(1)



def main(aid,title):

    for page in range(1,15):
        comment_url = f'https://api.bilibili.com/x/v2/reply?callback=jQueryjsonp=jsonp&pn={page}&type=1&oid={aid}&sort=2&_=1594459235799'

        response = send_get(comment_url,headers,params={})
        replies = response.get("data",{}).get("replies",[])
        if replies is None:
            replies = []
        for reply in replies:
            try:
                saveitem = {}
                saveitem["电影"] = title
                saveitem["视频id"] = aid
                saveitem["评论内容"] = reply.get("content", {}).get("message")
                saveitem["创建时间"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(reply.get("ctime")))
                saveitem["评论点赞数"] = reply.get("like")
                try:
                    saveitem["评论回复"] = "; ".join(
                        [i.get("content", {}).get("message", "") for i in reply.get("replies", [])])
                except:
                    saveitem["评论回复"] = ""
                print(conf["p"],title, page, saveitem)
                with open("comment.txt", 'a', encoding='utf-8') as f:
                    f.write(json.dumps(saveitem))
                    f.write('\n')
            except Exception as e:
                print(f"parse error comment:{e}")

if __name__ == '__main__':

    # records = pandas.read_excel("视频信息.xlsx",dtype=str).to_dict(orient='records')
    # conf = {"p":1051}
    # for record in records[conf["p"]:]:
    #     conf["p"]+=1
    #     aid = record.get("视频aid")
    #     title = record.get("电影名")
    #     main(aid=aid,title=title)

    with open("comment.txt",'r',encoding='utf-8') as f:
        comments = [json.loads(i.strip()) for i in f.readlines()]
    df = pandas.DataFrame(comments)
    # df["评论内容"] = df["评论内容"].apply(lambda x: str(x).encode('UTF-8', 'ignore').decode('UTF-8'))
    df.to_excel("评论下载.xlsx",index=False)