import json
import pprint
import time
import pandas
import redis
import requests

headers = {
    "Accept": 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    "Cookie": '_lxsdk_cuid=1896db100fec8-01656acf45c0ac-7e565474-1fa400-1896db100fec8; ci=264%2C%E6%B9%98%E6%BD%AD; Hm_lvt_703e94591e87be68cc8da0da7cbd0be2=1698169207,1698169814,1698205521,1699355445; WEBDFPID=96yz3z518u69553w0yu540y88vv7262w81yz02y6xw997958w2u3yw92-2014715504399-1699355503618UGGCCSUfd79fef3d01d5e9aadc18ccd4d0c95071198; token=AgH1JV18br9JPgN1cfPXtx2w01Yz41dsCP-zaY_UaXuUmh3Hsj5_OqqWXy7ZgfN2anuAJWhiJ5um-wAAAADKGwAAytB1tsjdZsE9pY_kgpbn-heTSY4FyrrFybtgJCFRj9HfP9BY0gBeYnXLWvCHiIaQ; _lxsdk=65B70930729211EEB4D9EB7BB0E0B55DA7EF99FB756E42FDAA6B4D0A14204205; Hm_lpvt_703e94591e87be68cc8da0da7cbd0be2=1699356426; _lxsdk_s=18ba97a473b-d5-dae-a35%7C%7C55',
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.203'
}


def send_get(url, headers, params):
    while 1:
        try:
            print(f">>> 访问：{url}")
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=(4, 5)
            )
            time.sleep(.3)
            return response
        except Exception as e:
            print(f"some error{e}")
            time.sleep(1)


def get_data(movie_id):

    if redis_con.hexists('movie_detail',movie_id):
        return print(f">>> 曾下载：{movie_id}")

    movie_detail_url = f'https://api.maoyan.com/mmdb/movie/v5/{movie_id}.json'
    movie_detail_response = send_get(movie_detail_url, headers, {}).json().get("data", {}).get("movie", {})

    actor_url = f'https://api.maoyan.com/mmdb/v9/movie/{movie_id}/celebrities.json'
    actor_response = send_get(actor_url,headers,{}).json()

    pf_url = f'https://npro.maoyan.com/api/movie/miniProgram/detail/movieBox.json?movieId={movie_id}&openId=&channelId=40005'
    pf_response = send_get(pf_url, headers=headers, params={}).json().get("data", {}).get("movieBoxList", [])

    saveitem = {}
    saveitem["电影id"] = movie_id
    saveitem["电影名称"] = movie_detail_response.get("nm")
    saveitem["电影介绍"] = movie_detail_response.get("dra")
    saveitem["电影封面图"] = movie_detail_response.get("img")
    saveitem["电影语言"] = movie_detail_response.get("oriLang")
    saveitem["电影类型"] = movie_detail_response.get("cat")
    saveitem["电影导演"] = movie_detail_response.get("dir")
    saveitem["电影持续时间"] = movie_detail_response.get("dur")
    saveitem["标签"] = movie_detail_response.get("scm")
    saveitem["电影上映时间"] = movie_detail_response.get("pubDesc")
    saveitem["语言"] = movie_detail_response.get("oriLang")
    saveitem["评分"] = movie_detail_response.get("sc")
    saveitem["主演"] = movie_detail_response.get("star")
    saveitem["上映地区"] = movie_detail_response.get("src")
    saveitem["评分人数"] = movie_detail_response.get("snum")
    saveitem["想看人数"] = movie_detail_response.get("wish")
    saveitem["观看过人数"] = movie_detail_response.get("watched")

    try:
        saveitem["主演"] = "; ".join([ f'{i.get("cnm","")}-{i.get("desc","")}' for i in actor_response.get("data",{}).get("list",[{}])[-1]])

    except Exception as e:
        saveitem["主演"] = ""

    allkeys = ['总票房', '分账票房', '总人次', '预测票房']
    for ik in allkeys:
        saveitem[ik] = ""

    for im in pf_response:
        if im.get("title") in allkeys:
            saveitem[im.get("title")] = im.get("valueDesc", "") + im.get("unitDesc", "")
    redis_con.hset("movie_detail",movie_id,json.dumps(saveitem))
    pprint.pprint(saveitem)


if __name__ == "__main__":

    redis_con = redis.Redis(db=13)
    # for year in range(2013,2019)[::-1]:
    #     filepath = f'./config/{year}.txt'
    #     with open(filepath, 'r', encoding='utf-8') as f:
    #         lines = [i.strip() for i in f.readlines() if i.strip() != '']
    #     for mid in lines:
    #         get_data(movie_id=mid)


    pandas.DataFrame([json.loads(i.decode()) for i in redis_con.hvals("movie_detail")]).to_excel("movieinfo.xlsx",index=False)