import datetime  ##时间格式化
import json  ##json解析
import os  ##文件路径操作
import random  ##获取随机数
import re  ##正则解析
import time  ##时间格式化
import pandas  ##数据整理导出
import pymongo
import requests  ##发送网络请求

requests.packages.urllib3.disable_warnings()


def get_true_time(t):
    samp_time = time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S")) - 8 * 60*60
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(samp_time))

class Gtwitter():
    search_api = 'https://twitter.com/i/api/graphql/NA567V_8AFwu0cZEkAAKcw/SearchTimeline'

    def __init__(self, max_page):
        self.records = []
        self.max_page = max_page

    def getblog(self, params):

        while 1:

            try:
                print(f">>> 可用cookie：{len(cookies)}")

                if len(cookies) == 0:
                    print(f'>>> 暂无可用cookie：')
                    time.sleep(10)
                    continue

                cookie = random.choice(cookies)

                sctoken = ''.join(re.findall(r'ct0=(.*?);', cookie)[:1])
                print(sctoken)
                headers = {
                    'Accept': '*/*',
                    'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
                    'Content-Type': 'application/json',
                    'Cookie': cookie,
                    'Referer': 'https://twitter.com/search?f=live&q=%22blackberry%22%20until%3A2023-06-29%20since%3A2018-06-23&src=typed_query',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.188',
                    'X-Csrf-Token': sctoken,
                    'X-Twitter-Active-User': 'yes',
                    'X-Twitter-Auth-Type': 'OAuth2Session',
                    'X-Twitter-Client-Language': 'en'
                }

                res = requests.get(
                    self.search_api,
                    headers=headers,
                    params=params,
                    timeout=(30, 40),
                    verify=False
                )
                if '"message":"Rate limit exceeded."' in res.text:
                    print(f'"message":"Rate limit exceeded."',res.text)
                    # cookies.remove(cookie)
                    continue
                if 'Authorization: Denied by access control' in res.text:
                    print(cookie)
                    print(f'Authorization: Denied by access control')
                    cookies.remove(cookie)
                    continue
                if '"Could not authenticate you"' in res.text:
                    print("Could not authenticate you")
                    cookies.remove(cookie)
                    continue
                if res.status_code != 200:
                    print(res.status_code, res.text)
                    continue
                return res.json()

            except Exception as e:

                print(f">>> parse error: {e}")

    def parse(self, data, page, keyword):
        try:
            tweets = \
                data.get("data").get("search_by_raw_query").get("search_timeline").get("timeline").get("instructions")[
                    0].get("entries")
        except Exception as e:
            print(f">>> error: {e}")
            print(data)
            return []
        if tweets is None:
            tweets = []
        print(page, len(tweets[:-2]))
        for it in tweets[:-2]:
            try:
                try:
                    legacy_handle = it['content']['itemContent']['tweet_results']['result']['legacy']
                    core_handle = it['content']['itemContent']['tweet_results']['result']['core']
                except:
                    legacy_handle = it['content']['itemContent']['tweet_results']['result']['tweet']['legacy']
                    core_handle = it['content']['itemContent']['tweet_results']['result']['tweet']['core']

                dt_obj = datetime.datetime.strptime(legacy_handle['created_at'], '%a %b %d %H:%M:%S %z %Y')
                a = core_handle['user_results']['result']['legacy']['screen_name']

                dic = {}
                dic['检索关键字'] = sub_key

                dic['用户名'] = core_handle['user_results']['result']['legacy']['name']
                dic['用户id'] = core_handle['user_results']['result']['legacy']['screen_name']
                dic['用户链接'] = f'https://twitter.com/{a}'
                dic['发布时间'] = get_true_time(dt_obj.astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S'))
                dic['发布内容'] = legacy_handle['full_text'].split("https://t.co/")[0]

                dic['博文id'] = legacy_handle['id_str']
                dic['评论数'] = legacy_handle["reply_count"]
                dic['转发数'] = legacy_handle["retweet_count"]
                dic['收藏数'] = legacy_handle["favorite_count"]
                dic['引用数'] = legacy_handle["quote_count"]
                dic["对某人的回复"] = legacy_handle.get("in_reply_to_screen_name", "")  #
                dic["文本长度"] = len(dic["发布内容"].split())
                dic["语言类型"] = legacy_handle['lang']

                try:
                    dic['图片列表'] = "; ".join(
                        [i.get("media_url_https") for i in legacy_handle.get("entities", []).get("media", []) if
                         i.get("type") == 'photo'])
                except:
                    dic['图片列表'] = ""

                try:
                    dic['视频封面图'] = "; ".join(
                        [i.get("media_url_https") for i in legacy_handle.get("entities", []).get("media", []) if
                         i.get("type") == 'video'])
                except:
                    dic['视频封面图'] = ""

                try:
                    dic['观看量'] = it['content']['itemContent']['tweet_results']['result']['views']["count"]
                except:
                    dic['观看量'] = 0

                try:
                    dic['标签'] = "; ".join([
                        i.get("text")
                        for i in legacy_handle["entities"]["hashtags"]
                    ])
                except Exception as e:
                    dic['标签'] = ""

                idd = legacy_handle['id_str']
                dic['博文链接'] = f'https://twitter.com/user/status/{idd}'

                quote_content = it['content']['itemContent']['tweet_results']['result'].get("quoted_status_result")

                if quote_content is not None:

                    dic["引文内容"] = quote_content.get("result",{}).get("legacy",{}).get("full_text").split("https://t.co")[0]

                else:
                    dic["引文内容"] = ""

                dic['发布者粉丝数'] = core_handle['user_results']['result']['legacy'].get("followers_count")
                dic['发布者关注数'] = core_handle['user_results']['result']['legacy'].get("friends_count")
                dic['发布者收藏数'] = core_handle['user_results']['result']['legacy'].get("favourites_count")
                dic['发布者ip属地'] = core_handle['user_results']['result']['legacy'].get("location")
                dic['发布者简介'] = core_handle['user_results']['result']['legacy'].get("description")
                dic['发布者生日'] = core_handle['user_results']['result']['legacy'].get("user_birthday")
                dic['发布者注册实际'] = core_handle['user_results']['result']['legacy'].get("created_at")
                dic['发布者媒体数'] = core_handle['user_results']['result']['legacy'].get("media_count")
                dic['发布者贴子数'] = core_handle['user_results']['result']['legacy'].get("statuses_count")
                dic['是否认证'] = core_handle['user_results']['result'].get("is_blue_verified")




                print(keyword, page, dic["发布时间"], dic)

                self.count += 1

                post_id = dic.get("博文id")
                dic["_id"] = post_id

                try:
                        pymongo_cur.insert_one(dic)
                except Exception as e:
                        print(e)
            except Exception as e:
                print(f'some error:{e}')
        return tweets

    def changev(self, text):
        try:
            dt_obj = datetime.datetime.strptime(text, '%a %b %d %H:%M:%S %z %Y')
            return dt_obj.astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S')

        except Exception as e:
            print(f">>> 时间格式异常： {text}")
            return text

    def blockdata(self, subkey,date1,date2):
        cursor = {"p": ''}
        self.count = 0
        for page in range(1, self.max_page):

            try:
                #  -filter:replies
                params = {
                    "fieldToggles": '{"withArticleRichContentState":false}',
                    "features": '{"rweb_lists_timeline_redesign_enabled":true,"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"creator_subscriptions_tweet_preview_api_enabled":true,"responsive_web_graphql_timeline_navigation_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"tweetypie_unmention_optimization_enabled":true,"responsive_web_edit_tweet_api_enabled":true,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,"view_counts_everywhere_api_enabled":true,"longform_notetweets_consumption_enabled":true,"responsive_web_twitter_article_tweet_consumption_enabled":false,"tweet_awards_web_tipping_enabled":false,"freedom_of_speech_not_reach_fetch_enabled":true,"standardized_nudges_misinfo":true,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":true,"longform_notetweets_rich_text_read_enabled":true,"longform_notetweets_inline_media_enabled":true,"responsive_web_media_download_video_enabled":false,"responsive_web_enhance_cards_enabled":false}',
                    "variables": r'{"rawQuery":"\"xxxx1\" lang:en until:xxxx2 since:xxxx3 -filter:replies","cursor":"' + cursor[
                        "p"] + r'","count":20,"querySource":"typed_query","product":"Latest"}'
                }
                params["variables"] = params["variables"].replace('xxxx1', subkey)
                params["variables"] = params["variables"].replace('xxxx2', date2)
                params["variables"] = params["variables"].replace('xxxx3', date1)

                blogInfo = self.getblog(params)

                tweets = self.parse(blogInfo, page, subkey)
                pagelength = len(tweets) - 2
                if 1:
                    if page == 1 and cursor["p"] == "":

                        cursor["p"] = tweets[-1].get("content").get("value")
                    else:
                        instructions = \
                            blogInfo.get("data").get("search_by_raw_query").get("search_timeline").get("timeline").get(
                                "instructions")[-1]
                        cursor["p"] = instructions.get("entry").get("content").get("value")
                    print(f"下一页：{cursor['p']}")
                if cursor["p"] is None or pagelength < 1:
                    print('暂无下一页：')
                    break
            except Exception as e:
                print(f"eroro{e}")
                break

    @staticmethod
    def getdaterange(starttime, endtime):
        time_str = []
        start_samp = time.mktime(time.strptime(starttime, '%Y-%m-%d'))
        while True:
            time_str.append(
                (time.strftime('%Y-%m-%d', time.localtime(start_samp)),
                 time.strftime('%Y-%m-%d', time.localtime(start_samp + 24 * 60 * 60 * 1)))
            )
            start_samp += 24 * 60 * 60 * 1
            if start_samp > time.mktime(time.strptime(endtime, '%Y-%m-%d')):
                break
        return time_str[::-1]


def parse_cookie():
    with open("./config/newcookie.txt", 'r', encoding='utf-8') as f:
        lines = [i.strip() for i in f.readlines()]
        for line in lines:

            if len([i for i in line.split("|") if 'ct0' in i]) >= 0:

                try:
                    cookies.append([i.strip() for i in line.split("|") if 'ct0' in i][0])
                    print(f"parse cookie:", [i.strip() for i in line.split("|") if 'ct0' in i][0])
                except:
                    pass

def parse_cookie2():
    with open("./config/newcookie2.txt", 'r', encoding='utf-8') as f:
        lines = [i.strip() for i in f.readlines()]
        for line in lines:

                try:

                    sub_lines = line.split(";")
                    auth_token = sub_lines[3]
                    ct0 = sub_lines[4]
                    cookiestr = f'auth_token={auth_token};lang=en;ct0={ct0};'
                    print(cookiestr)
                    cookies.append(cookiestr)

                except:
                    pass


if __name__ == "__main__":
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    cookies = []
    parse_cookie2()

    start_date = '2022-01-01'
    end_date = '2024-12-31'

    pymongo_con = pymongo.MongoClient("mongodb://localhost:27017/")
    pymongo_cur = pymongo_con["twitter-btc"]['data']


    for sub_key in [
        'bitcoin'
    ]:

        for date1,date2 in Gtwitter.getdaterange(start_date,end_date)[::-1]:

            tt = Gtwitter(max_page=60)
            tt.blockdata(sub_key,date1,date2)


    df = pandas.DataFrame([i for i in pymongo_cur.find({})])
    df.drop_duplicates(['博文id'], inplace=True)
    with pandas.ExcelWriter('post.xlsx', engine='xlsxwriter', options={'strings_to_urls': False}) as writer:
        df.to_excel(writer, index=False)
