import datetime
import json
import math
import os
import random
import re
import time
import pandas
import pymongo
import requests
from x_client_transaction import ClientTransaction

digits = "0123456789abcdefghijklmnopqrstuvwxyz"

def baseConversion(x, base):
    result = ''
    i = int(x)
    while i > 0:
        result = digits[i % base] + result
        i = i // base
    if int(x) != x:
        result += '.'
        i = x - int(x)
        d = 0
        while i != int(i):
            result += digits[int(i * base % base)]
            i = i * base
            d += 1
            if d >= 8:
                break
    return result


def calcSyndicationToken(idStr):
    id = int(idStr) / 1000000000000000 * math.pi
    o = baseConversion(x=id, base=int(math.pow(6, 2)))
    c = o.replace('0', '').replace('.', '')
    if c == '':
        c = '0'
    return c

def get_twitter_homepage(headers=None):
    if headers is None:
        headers = {"Authority": "x.com",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Referer": "https://x.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "X-Twitter-Active-User": "yes",
            "X-Twitter-Client-Language": "en"}
    if 'Authorization' in headers:
        del headers['Authorization']
    response = requests.get("https://x.com/home", headers=headers)
    return response

def generate_transaction_id(method: str, path: str,headers=None) -> str:
    ct = ClientTransaction(get_twitter_homepage(headers=headers))
    transaction_id = ct.generate_transaction_id(method=method, path=path)
    return transaction_id



class Gtwitter():
    search_api = 'https://x.com/i/api/graphql/JpDTssKfMuPsqyee8-TDQA/Retweeters'
    resulst = []

    def __init__(self, max_page):
        self.max_page = max_page

    def getblog(self, params):

        while 1:

            try:
                if len(cookies) <= 0:
                    input(f"空 cookie")


                endpoint = f"/i/api/graphql/JpDTssKfMuPsqyee8-TDQA/Retweeters"
                transaction_id = generate_transaction_id("GET", endpoint)


                cookie = random.choice(cookies)
                print(f">>> 可用cookie数：{len(cookies)}")
                sctoken = ''.join(re.findall(r'ct0=(.*?);', cookie)[:1])
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
                    'X-Twitter-Client-Language': 'en',
                    "x-client-transaction-id": transaction_id,

                }

                res = requests.get(
                    self.search_api,
                    headers=headers,
                    params=params,
                    timeout=(3, 4)
                )
                if 'Rate limit exceeded' in res.text:
                    print(f'{flag} Rate limit exceeded')
                    # cookies.remove(cookie)
                    continue
                if 'Authorization: Denied by access control' in res.text:
                    print(f'{flag} Authorization: Denied by access control')
                    cookies.remove(cookie)
                    continue
                if res.status_code != 200:
                    print(f'{flag} ', res.status_code, res.text)
                    cookies.remove(cookie)
                    continue
                return res.json()

            except Exception as e:

                print(f"{flag} >>> parse error: {e}")
                time.sleep(1)

    def parse(self, response, page, focalTweetId,twitter_user_id):
        try:
            tweets = response.get("data").get("retweeters_timeline").get("timeline").get("instructions")[0].get("entries")
        except Exception as e:
            print(f"{flag} >>> error: {e}")
            time.sleep(1)
            return []
        if tweets is None:
            tweets = []
        for it in tweets:
            try:

                legacy = it["content"]["itemContent"]["user_results"]["result"]["legacy"]

                dic = {}
                dic["_id"]= it.get("sortIndex")
                dic["博文id"] = str(focalTweetId)
                dic["博文发布者"] = twitter_user_id
                dic["转发id"] = it.get("sortIndex")

                dic['转发者名称'] = legacy.get("name")
                dic['转发者id'] = legacy.get("screen_name")
                dic['转发者粉丝数'] = legacy.get("followers_count")
                dic['转发者关注数'] = legacy.get("friends_count")
                dic['转发者收藏数'] = legacy.get("favourites_count")
                dic['转发者ip属地'] = legacy.get("location")
                dic['转发者简介'] = legacy.get("description")
                dic['转发者生日'] = legacy.get("user_birthday")
                dic['转发者注册实际'] = legacy.get("created_at")
                dic['转发者媒体数'] = legacy.get("media_count")
                dic['转发者贴子数'] = legacy.get("statuses_count")

                print(f'{flag} ', page, dic)

                self.comment_count += 1
                try:
                    pymongo_cur.insert_one(dic)
                except Exception as e:
                    print(f'{flag} >>> insert error: {e}')


            except Exception as e:
                print(f'some error:{e}')
        return tweets

    def changev(self, text):
        try:
            dt_obj = datetime.datetime.strptime(text, '%a %b %d %H:%M:%S %z %Y')
            return dt_obj.astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S')

        except Exception as e:
            print(f"{flag}>>> 时间格式异常： {text}")
            return text

    def blockdata(self, twitter_url,twitter_user_id):

        self.comment_count = 0
        focalTweetId = twitter_url.split("status/")[-1]

        print(f"{flag}正在访问的博文序号：", focalTweetId)
        cursor = {"p": ''}
        for page in range(1, self.max_page):

            try:
                params = {
                    "variables": '{"tweetId":"' + str(focalTweetId) + '","count":20,"cursor":"' + str(cursor["p"]) + '","includePromotedContent":true}',
                    "features": '{"rweb_video_screen_enabled":false,"profile_label_improvements_pcf_label_in_post_enabled":true,"rweb_tipjar_consumption_enabled":true,"verified_phone_label_enabled":false,"creator_subscriptions_tweet_preview_api_enabled":true,"responsive_web_graphql_timeline_navigation_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"premium_content_api_read_enabled":false,"communities_web_enable_tweet_community_results_fetch":true,"c9s_tweet_anatomy_moderator_badge_enabled":true,"responsive_web_grok_analyze_button_fetch_trends_enabled":false,"responsive_web_grok_analyze_post_followups_enabled":true,"responsive_web_jetfuel_frame":false,"responsive_web_grok_share_attachment_enabled":true,"articles_preview_enabled":true,"responsive_web_edit_tweet_api_enabled":true,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,"view_counts_everywhere_api_enabled":true,"longform_notetweets_consumption_enabled":true,"responsive_web_twitter_article_tweet_consumption_enabled":true,"tweet_awards_web_tipping_enabled":false,"responsive_web_grok_show_grok_translated_post":false,"responsive_web_grok_analysis_button_from_backend":true,"creator_subscriptions_quote_tweet_preview_enabled":false,"freedom_of_speech_not_reach_fetch_enabled":true,"standardized_nudges_misinfo":true,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":true,"longform_notetweets_rich_text_read_enabled":true,"longform_notetweets_inline_media_enabled":true,"responsive_web_grok_image_annotation_enabled":true,"responsive_web_enhance_cards_enabled":false}',
                }
                response = self.getblog(params)

                tweets = self.parse(response, page, focalTweetId,twitter_user_id)
                print(f'>>> last',tweets[-1])

                if tweets[-1].get("content").get("value") == cursor["p"]:
                    print(f">>> 游标重复：",cursor["p"])
                    break

                cursor["p"] = tweets[-1].get("content").get("value")
                if cursor["p"] is None:
                    print(f">>> 暂无下一页：",cursor["p"])
                    break
            except Exception as e:
                print(f">>> 暂无下一页：{e}")
                break


def getdaterange(starttime, endtime):
    time_str = []
    start_samp = time.mktime(time.strptime(starttime, '%Y-%m-%d'))
    while True:
        time_str.append(time.strftime('%Y-%m-%d', time.localtime(start_samp)))
        start_samp += 24 * 60 * 60
        if start_samp > time.mktime(time.strptime(endtime, '%Y-%m-%d')):
            break
    return time_str


def parse_cookie2():
    with open("./config/newcookie2.txt", 'r', encoding='utf-8') as f:
        lines = [i.strip() for i in f.readlines()]
        for line in lines:

                try:
                    cookies.append(line)
                    # sub_lines = line.split(";")
                    # auth_token = sub_lines[3]
                    # ct0 = sub_lines[4]
                    # cookiestr = f'auth_token={auth_token};lang=en;ct0={ct0};'
                    # print(cookiestr)
                    # cookies.append(cookiestr)
                except:
                    pass

if __name__ == "__main__":
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    cookies = []

    parse_cookie2()

    pymongo_con = pymongo.MongoClient("mongodb://localhost:27017/")
    pymongo_cur = pymongo_con["twitter-olympic"]['repost']

    all_post = [

        i
        for i in pandas.read_excel("post.xlsx", dtype=str).to_dict(orient="records")

        if int(i.get("转发数")) >= 10
    ]

    # count_current_blog = 10833
    #
    # for twitter_info in all_post[count_current_blog:]:
    #     count_current_blog += 1
    #     flag = f'进度【{count_current_blog}/{len(all_post)}】'
    #
    #     tt = Gtwitter(10000)
    #     tt.blockdata(twitter_info.get("博文链接"),twitter_info.get("用户id"))

    df = pandas.DataFrame([i for i in pymongo_cur.find({})])
    df.to_csv("repost.csv",index=False)
    # with pandas.ExcelWriter('repost.csv', engine='xlsxwriter', options={'strings_to_urls': False}) as writer:
    #     df.to_excel(writer, index=False)
