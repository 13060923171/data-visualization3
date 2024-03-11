import base64
import datetime
import json
import os
import random
import re
import time
import copyheaders
import langid
import pandas
import pymongo
import requests


class Gtwitter():
    search_api = 'https://twitter.com/i/api/graphql/g-nnNwMkZpmrGbO2Pk0rag/TweetDetail'
    resulst = []

    def __init__(self, max_page):
        self.max_page = max_page

    def getblog(self, params):

        while 1:

            try:

                cookie = random.choice(cookies)

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
                    'X-Twitter-Client-Language': 'en'
                }

                res = requests.get(
                    self.search_api,
                    headers=headers,
                    params=params,
                    timeout=(3, 4)
                )
                if 'Rate limit exceeded' in res.text:
                    print(f'{flag} Rate limit exceeded')
                    time.sleep(2)
                    continue
                if 'Authorization: Denied by access control' in res.text:
                    print(f'{flag} Authorization: Denied by access control')
                    time.sleep(2)
                    cookies.remove(cookie)
                    continue
                if res.status_code != 200:
                    print(f'{flag} ', res.status_code, res.text)
                    time.sleep(1)
                    continue
                return res.json()

            except Exception as e:

                print(f"{flag} >>> parse error: {e}")
                time.sleep(1)

    def parse(self, response, page, blog):
        try:
            tweets = response.get("data").get("threaded_conversation_with_injections_v2").get("instructions")[0].get(
                "entries")
        except Exception as e:
            print(f"{flag} >>> error: {e}")
            time.sleep(1)
            return []
        if tweets is None:
            tweets = []
        for it in tweets:
            try:

                legacy = it["content"]["items"][0]["item"]["itemContent"]["tweet_results"]["result"]
                dic = blog.copy()
                dic["评论者id"] = legacy["legacy"].get("user_id_str")
                dic["评论者名称"] = legacy["core"].get("user_results").get("result").get("legacy").get("screen_name")
                dic["评论id"] = legacy["legacy"].get("id_str")
                dic["评论内容"] = legacy["legacy"].get("full_text").split("https://t.co/")[0]
                print(f'{flag} ', page, dic["评论内容"].replace("\r", "").replace("\n", "").replace("\t", ""))

                self.comment_count += 1
                current_blog.append(dic)


            except Exception as e:
                pass
        return tweets

    def changev(self, text):
        try:
            dt_obj = datetime.datetime.strptime(text, '%a %b %d %H:%M:%S %z %Y')
            return dt_obj.astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S')

        except Exception as e:
            print(f"{flag}>>> 时间格式异常： {text}")
            return text

    def blockdata(self, blog):
        self.comment_count = 0
        focalTweetId = blog.get("博文id")

        print(f"{flag}正在访问的博文序号：", focalTweetId, f'该博文总评论数：{blog["回复数量"]}')
        cursor = {"p": ''}
        for page in range(1, self.max_page):

            try:
                params = {
                    "fieldToggles": '{"withArticleRichContentState":false}',
                    "features": '{"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"responsive_web_home_pinned_timelines_enabled":true,"creator_subscriptions_tweet_preview_api_enabled":true,"responsive_web_graphql_timeline_navigation_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"tweetypie_unmention_optimization_enabled":true,"responsive_web_edit_tweet_api_enabled":true,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,"view_counts_everywhere_api_enabled":true,"longform_notetweets_consumption_enabled":true,"responsive_web_twitter_article_tweet_consumption_enabled":false,"tweet_awards_web_tipping_enabled":false,"freedom_of_speech_not_reach_fetch_enabled":true,"standardized_nudges_misinfo":true,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":true,"longform_notetweets_rich_text_read_enabled":true,"longform_notetweets_inline_media_enabled":true,"responsive_web_media_download_video_enabled":false,"responsive_web_enhance_cards_enabled":false}',
                    "variables": '{"focalTweetId":"' + str(focalTweetId) + '","cursor":"' + str(cursor[
                                                                                                    "p"]) + '","referrer":"tweet","with_rux_injections":false,"includePromotedContent":true,"withCommunity":true,"withQuickPromoteEligibilityTweetFields":true,"withBirdwatchNotes":true,"withVoice":true,"withV2Timeline":true}'
                }
                response = self.getblog(params)

                tweets = self.parse(response, page, blog)
                if 1:
                    cursor["p"] = tweets[-1].get("content").get("itemContent").get("value")
                if cursor["p"] is None:
                    break
            except Exception as e:
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


def check_language(string: str) -> str:
    """检查语言
    :return zh:中文,en:英文,
    """
    if string == "":
        return 'en'

    new_string = re.sub(r'[0-9]+', '', string)  # 这一步剔除掉文本中包含的数字
    return langid.classify(new_string)[0]


if __name__ == "__main__":

    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    ###账号的cookie，需要买账后登录后复制在这里作为列表的元素
    cookies = [
        'ct0=5e7bd4ae3018934df3a4aaf03caaf92cfc8adac3ae5c7d45581f4dc1face5af7b93d7a6442efcffe4caafc80dc3671285c8c2c326fe75d79887224d23471dd14683dbfcd949423c8e250accce21c96a3; guest_id_marketing=v1%3A169441283378026239; guest_id_ads=v1%3A169441283378026239; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCPcD4IKKAToMY3NyZl9p%250AZCIlNmI1MTAwOWQ4OWU2NGE5ZjYwZTQ1OWEyOTg0YWI1Y2I6B2lkIiVkYTQ5%250AYTQzODBhMDk4MzU2MDUzOTBiM2QxYzg2YjcwNA%253D%253D--f90441c5710806cc92da782e8d6e9424a377cbb1; personalization_id="v1_tvAqBgp8gbt08b4gYmA1Bg=="; guest_id=v1%3A169441283378026239; kdt=XSQocHE5t4CYxrYgPtXOhljzNrSH0ZQ4abR58Sdj; twid="u=1701116888867106816"; auth_token=fa6ce9b7a2d4bd31a7c44bfbcbb6a430cc6acabb',
        'guest_id_marketing=v1%3A169166115591780722; guest_id_ads=v1%3A169166115591780722; personalization_id="v1_9o9FhJK39A9waXIP7XYN2A=="; guest_id=v1%3A169166115591780722; ct0=8f9d63b4b4b676158dea3571a419114a95e348209a80b1aaefb0714d4220c2681673b80e32206eff68feff0e1ba12507fd3c1b8c084171ea2d267b237ef67bc2aec28d5e1a660d1c916a99878d83d7d3; _ga=GA1.2.505451109.1691661156; kdt=YQKWYPd1AHeddxUAK2mD7nioOufdDtpSBvXNbvCJ; twid=u%3D1677970978469711878; auth_token=ab4c712609e09a17098791f771653078bbcb6ffe; lang=en; _gid=GA1.2.1871236937.1694872179',
        '_ga=GA1.2.544820069.1694872147; _gid=GA1.2.237328157.1694872147; guest_id=v1%3A169487214989525801; guest_id_marketing=v1%3A169487214989525801; guest_id_ads=v1%3A169487214989525801; gt=1703044636997280148; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCLNXRZ6KAToMY3NyZl9p%250AZCIlN2UxNTA1ODNjMDNhOGIwZWRlZTE4ODYxZDQ2MTIxOWU6B2lkIiVlOWNj%250AOTcxZThiYzBmNDBhYzI4OWQ5NDU1MWUzNzBiOQ%253D%253D--b824d994701cb0d993efa658b08adafe85c58fef; kdt=cuXBIcdowS9AFMKCuKSWftV5eZRLmt6ttbQNiths; auth_token=5e607f7cb3c3bf4dbc4079378c77a4c65f5e49bd; ct0=95054246d19d09408e6eab5146e62387138f7782d2e2da3c3ce20c495c06f983d7a2670aaf4a596239e41e5cc7cf9095180de744a3bc98395c18417158734690d790cad3d070f23561ae1ae1e4b3f031; lang=en; twid=u%3D1677960686872465409; personalization_id="v1_nNPCT5jjZr4i/pIS0e0qnQ=="',
        '_ga=GA1.2.544820069.1694872147; _gid=GA1.2.237328157.1694872147; gt=1703044636997280148; kdt=cuXBIcdowS9AFMKCuKSWftV5eZRLmt6ttbQNiths; dnt=1; auth_multi="1677960686872465409:5e607f7cb3c3bf4dbc4079378c77a4c65f5e49bd"; auth_token=5ac9aa94b863a6495d3982d28a7da7313e35a629; guest_id=v1%3A169487259833497962; ct0=e702b4699242025fb6f1bcaf96d1b2f1bce9a2eedd883b2a0f9d74b702fb2334deaac0b333bfcb750c0ed45e77915422911a69ad590718240aa2764dd57510aea880578964510a810a6559173bf843af; guest_id_ads=v1%3A169487259833497962; guest_id_marketing=v1%3A169487259833497962; personalization_id="v1_CRpnNauyVPs0BmQpWqF02Q=="; twid=u%3D1677969021592043521',
        '_ga=GA1.2.544820069.1694872147; _gid=GA1.2.237328157.1694872147; gt=1703044636997280148; kdt=cuXBIcdowS9AFMKCuKSWftV5eZRLmt6ttbQNiths; lang=en; dnt=1; auth_multi="1677969021592043521:5ac9aa94b863a6495d3982d28a7da7313e35a629|1677960686872465409:5e607f7cb3c3bf4dbc4079378c77a4c65f5e49bd"; auth_token=e96012645b95e5941663ae0b39cb2756d2f2d398; guest_id=v1%3A169487267552511573; ct0=dc68f1fd2150201aefdf0c0794b16934de5df4d4660a72f64112d13aa9aeb208c6c3dbf21acfa52c986374daf8f59dba79445be45d8729c15ae8a8c296baea019c1366744fb97d01f2fdae927cf9872e; guest_id_ads=v1%3A169487267552511573; guest_id_marketing=v1%3A169487267552511573; twid=u%3D1677970978469711878; personalization_id="v1_sHhiilAUsqeXLxF4Vbsh1Q=="',
        '_ga=GA1.2.544820069.1694872147; _gid=GA1.2.237328157.1694872147; gt=1703044636997280148; kdt=cuXBIcdowS9AFMKCuKSWftV5eZRLmt6ttbQNiths; dnt=1; auth_multi="1677970978469711878:e96012645b95e5941663ae0b39cb2756d2f2d398|1677969021592043521:5ac9aa94b863a6495d3982d28a7da7313e35a629|1677960686872465409:5e607f7cb3c3bf4dbc4079378c77a4c65f5e49bd"; auth_token=828093581af38971295a349235938e253c9e6d7a; guest_id=v1%3A169487276783555871; ct0=6c4129933a80b80274671bfb5c37488194f1e26b824186b6f514b8dda10c0fc03d1ae6f2204d5c1350a3c851f5c1e534a1124a02f52da420d43616ba4585542d4ee4a1d90ac5e671f78d3e4dff4c0f2c; guest_id_ads=v1%3A169487276783555871; guest_id_marketing=v1%3A169487276783555871; twid=u%3D1677932686739120128; personalization_id="v1_7XX9lX+LDpkoOUhfXslgOA=="',
        '_ga=GA1.2.544820069.1694872147; _gid=GA1.2.237328157.1694872147; gt=1703044636997280148; kdt=cuXBIcdowS9AFMKCuKSWftV5eZRLmt6ttbQNiths; dnt=1; auth_multi="1677932686739120128:828093581af38971295a349235938e253c9e6d7a|1677970978469711878:e96012645b95e5941663ae0b39cb2756d2f2d398|1677969021592043521:5ac9aa94b863a6495d3982d28a7da7313e35a629|1677960686872465409:5e607f7cb3c3bf4dbc4079378c77a4c65f5e49bd"; auth_token=590bb7a85b220b6e7ffd032d06457459bb24ce90; guest_id=v1%3A169487380403473493; ct0=56a59709da02c98ae54e1fdf68fb099eacc5d6ae2f77269560570512f8ea1916caac1b9f9d98ad4444ddd21c97aea719c4c0b4c76afc508ce50d8f633ca6e0afb0e4096727e77df4a7774688348f495a; guest_id_ads=v1%3A169487380403473493; guest_id_marketing=v1%3A169487380403473493; twid=u%3D1701237556946333697; personalization_id="v1_f7RY+RIps+0vvrsNUGBDkw=="',
        'ct0=44a33b7c496fb4d8335ea0bf84de6defaf7a7c3039c24b3fdf830efbd991d219159041d0d2980fa3af83a62979003512b1be432eddb428a075e93d337f2b413203b32b63c2f9b55cb89a5e8469295cdc; guest_id_marketing=v1%3A169270740614619225; guest_id_ads=v1%3A169270740614619225; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCERBOR2KAToMY3NyZl9p%250AZCIlMjQ1ZGQ2ZWZkMjZkNTdkYTEzM2ZhOTEwNWRhZjU2NTc6B2lkIiVkNDhj%250ANDYzNTdmOWY1MTE1Yjc1NWUyN2Q4ZDg2N2MyNg%253D%253D--2a4cfdd960229219f0a30f55c6bf18a7b557aa8b; personalization_id="v1_WXACyQi7vScqB1cWoVmg3g=="; guest_id=v1%3A169270740614619225; kdt=TzyHerqZ1oiFH2lJ6fCUS9D1EdlVw8hKopmxqkKo; twid="u=1693963826822070272"; auth_token=35b2133de68351647b7ac91dc0e8060d12cde547',
        'ct0=4890661f00dd8c126a00f8cb166767834fc7185befc725b0001648d82843fbda6684be6080d8601c44eb5adde7dc422a6e26007779494961a9ecac16fd560d803a380270dc0e30645d358688f6f00b78; guest_id_marketing=v1%3A169270739055487428; guest_id_ads=v1%3A169270739055487428; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCF4EOR2KAToMY3NyZl9p%250AZCIlYmVkZjc1OGEzOGQwMDYxMDA3ZWI3ODgyYWQ2NjlmYzA6B2lkIiViYjIw%250AMzkwODBkNzA1YmQ5MzRlNGZhNGI5YjQzNGJjYQ%253D%253D--81c17d78acb40a9e672a8372bcd0ca33d69dc8c8; personalization_id="v1_JyO9fao+SxbXnRA6g9Mh6w=="; guest_id=v1%3A169270739055487428; kdt=3n2U54IGfWhnIXO03Bu0bBw5T3lL3tzso3mb3pSg; twid="u=1693963761990987776"; auth_token=9866a35b4ac4dfa578f707974ccac4e6591d7d09',
        'ct0=ff303e6fae71b23b26ddad90c4655f7465e29e4dd2c213ed35d29da52ff96154153de5b9082dcc2b3e2bb0bfa3af66cee001c4b9d03fe0cc5a39a458c763fd5df02f19d2f69dfd938409b3fc4d4432e3; guest_id_marketing=v1%3A169270738205367590; guest_id_ads=v1%3A169270738205367590; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCCjjOB2KAToMY3NyZl9p%250AZCIlYjA0NDkxMzFiOWVmMDdlYzVhZjYxOTNlZDU0ZWQyYTU6B2lkIiU1NjM0%250AZDY5MGZhN2RkYTg3ODc5YmVhNzI0NDEzYTk5Ng%253D%253D--b9ad78a68f8b061e6f0f54311a1bb98dbb05ba7c; personalization_id="v1_U0mT5HoBawRQJTD0IjvhEg=="; guest_id=v1%3A169270738205367590; kdt=YJTfLxgUtqpDTlObHj5csZ96rqiO1YX8Y5RiRPOs; twid="u=1693963705979944961"; auth_token=84a85014ab3e190a02b4116528cca88859e02870',
        'ct0=34660933a67870b67f93c4f03c055a7dcc3224bb23017b838c68deef3bcf8a2c7062c49ef2b3e2efc4f6c9a726885ba23f6cdd2cd237c2fc263b1f260765bb3f72d2cb9508baca709650cdd31aba7124; guest_id_marketing=v1%3A169270735486186461; guest_id_ads=v1%3A169270735486186461; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCPB4OB2KAToMY3NyZl9p%250AZCIlYTk3MTFlMTliYTcyNzgxMGQ1NDU2N2YwM2FjOGZhMWY6B2lkIiU5Mjlk%250AYjg0MjIyZTJkZjM5ZjdhMTMzNzZlM2EyNDg2MQ%253D%253D--480e05ce215a47c099e88cbf96d1175a44faa505; personalization_id="v1_Bmi7JsF1aEO9XEi/bABRgQ=="; guest_id=v1%3A169270735486186461; kdt=Ix6necP3dTCEpl6Tnx3vifHi2cKynGclbXGgwUEh; twid="u=1693963639542394880"; auth_token=0ea45a2fdc5ccb957b1043e5fe5855c50da1d6a5',
        'ct0=61abc8fd2937fc871b44839674c1260edc9f93f82b74a5be39d588887390461b6073b0d8f01298a320f0b848375a5b64f57ce15b712cc607b5ad2cdff05fbee7d465fc79ee36da4df2df5f1fcf8ba38b; guest_id_marketing=v1%3A169270735388062274; guest_id_ads=v1%3A169270735388062274; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCB11OB2KAToMY3NyZl9p%250AZCIlMWM3ZjY0ODc1MDAzZDEzMmRiNTUwZDdjZTRkMmZmN2E6B2lkIiVhYTE0%250AYjk0ZDE1NWQzOGVjYmQyYzhmMDQ1NzZmNGQwYQ%253D%253D--c723c50441965c27b1fec7932efd7655dd890eb3; personalization_id="v1_u4XZr0kGw7HX5odeygzrIQ=="; guest_id=v1%3A169270735388062274; kdt=ZeKpbgHL2TxLDnYST3vInIQDl7Kvx1rnCGUnAItu; twid="u=1693963607107624960"; auth_token=0390fe9e9740bf143d9530a633ca697bae8a462a',
        '_ga=GA1.2.2001245201.1696121387; kdt=rSVZPNi76RnCpYJ6XHsFr4mHus3Dnfk7kQWtexwk; dnt=1; _gid=GA1.2.805936431.1696667554; auth_multi="1687314223993417728:90ab2e45dcad4788f145de14d33718416520ecf0|1687317782394724352:5e7029561a087d25cee416b5a8183e3e7b48f3fb"; auth_token=ce320b29d04769ef0ba65c782288c36e3c27d21d; guest_id=v1%3A169666874386661403; ct0=49b5ca2d82a386c4c8b3ae8752428a38430813cdbd1808cf2b87f8ab815e43600e5978e9c0e3caa4b1f6c5f699f6e784d7d0bc76c0b462d08c6e8fc91340713a72280b8c3ddb8f1206e037b3125a0728; guest_id_ads=v1%3A169666874386661403; guest_id_marketing=v1%3A169666874386661403; twid=u%3D1706097779951718400; personalization_id="v1_H4Ck5oHiQ8ABhtdhg0Z2lA=="',
        '_ga=GA1.2.2001245201.1696121387; kdt=rSVZPNi76RnCpYJ6XHsFr4mHus3Dnfk7kQWtexwk; lang=en; _gid=GA1.2.805936431.1696667554; dnt=1; auth_multi="1706097779951718400:ce320b29d04769ef0ba65c782288c36e3c27d21d|1687314223993417728:90ab2e45dcad4788f145de14d33718416520ecf0|1687317782394724352:5e7029561a087d25cee416b5a8183e3e7b48f3fb"; auth_token=59edced0a2e0aeff6639c8ddefcd35a57b322f06; guest_id=v1%3A169666884389909009; ct0=d51bfd797c3b7dfdb5a141af3d994726d985dbae8dbdb6c7daceb4303ddb71a85a63c59d8e7d9081b8eb5c3e2efb2c3047b5e46b0dd74b875cc353692f31c03aff90710b7c4a78763998f343d720b5ff; guest_id_ads=v1%3A169666884389909009; guest_id_marketing=v1%3A169666884389909009; twid=u%3D1706099101279076352; personalization_id="v1_izMR3ujDGn+//S8IMNCMRQ=="',
        '_ga=GA1.2.2001245201.1696121387; kdt=rSVZPNi76RnCpYJ6XHsFr4mHus3Dnfk7kQWtexwk; lang=en; _gid=GA1.2.805936431.1696667554; dnt=1; auth_multi="1706099101279076352:59edced0a2e0aeff6639c8ddefcd35a57b322f06|1706097779951718400:ce320b29d04769ef0ba65c782288c36e3c27d21d|1687314223993417728:90ab2e45dcad4788f145de14d33718416520ecf0|1687317782394724352:5e7029561a087d25cee416b5a8183e3e7b48f3fb"; auth_token=3768386c2e7114e1d83e17a626005ed1b0feb084; guest_id=v1%3A169666900048772979; ct0=a67a55e9dffc37ff15d159e5803643e3eaa63b585bac0028dae7d22bd99222176f497b4f0b8858b40e35b902aaeb5e9ead1c0b11fb807b4f12c4bbb840f50679c3ab45688e41ff1f1d416adb09779809; guest_id_ads=v1%3A169666900048772979; guest_id_marketing=v1%3A169666900048772979; personalization_id="v1_iPLOR6QU2SCudt3ShFKdYA=="; twid=u%3D1708403691794173952',
        'guest_id=v1%3A169666906924749250; _ga=GA1.2.1485958329.1696669070; _gid=GA1.2.2037767557.1696669070; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCCx%252BWwmLAToMY3NyZl9p%250AZCIlM2U3NmNhZGRmYjNmZjkxOTdjNjAwMTA2Y2Q2ODAzYzM6B2lkIiU0NTUw%250AODYzZmIzMDAyNGEzM2QxOGFkODBkYThhMWVmMw%253D%253D--7439d0d8586fcf498eb032c92914d8eb532ee617; kdt=WPWphveELWLH8bHoQt0RiHe8HnnUDFSwpDYDGOnd; auth_token=d455bdb1e943c768dd00148cc5feeb643db9e051; ct0=cd4f7eebcc886a4c7d0e12b7cfdbf739dc692d99620794eebdbca1746fd08e71203c434d5fa6f54e53a92985b296a26d8d629881d488a56c76a19f230ec2eba3593416c020bd1c6e6f691acbb5a77624; guest_id_ads=v1%3A169666906924749250; guest_id_marketing=v1%3A169666906924749250; lang=en; personalization_id="v1_t5YyELb6QnJiQ1Nz6tu1vA=="; twid=u%3D1706148813155762176',
        '_ga=GA1.2.1485958329.1696669070; _gid=GA1.2.2037767557.1696669070; kdt=WPWphveELWLH8bHoQt0RiHe8HnnUDFSwpDYDGOnd; lang=en; dnt=1; auth_multi="1706148813155762176:d455bdb1e943c768dd00148cc5feeb643db9e051"; auth_token=1e1ba37364a69f519de25944798d223192ccdf48; guest_id=v1%3A169666921455541788; ct0=a7cc61bb9a629335da7b873c530650b563e0f8b2d41f83d628e35a77257bee9c4cd9361ef23d1fe82f484a7480710353911601d6fa264201cc9bc8211e23437eecbb21dfdfdcbc7274ea234f2537bb34; guest_id_ads=v1%3A169666921429046044; guest_id_marketing=v1%3A169666921429046044; personalization_id="v1_YHxPKNY2fgUQitAD7H6lPQ=="; twid=u%3D1708460456044789760',
        'kdt=o2CdPi6Dj7OoySFXQo2Hxk3QEL9VqE5OmJLGMn4a; _ga=GA1.2.2142299479.1705300655; _gid=GA1.2.1824585528.1709186858; dnt=1; auth_multi="1442379882294484994:b240e37a558f84f433fd4cfaba7b09f976757e8a|1687316798842994688:1a3a4d63a39ab3d92e4f3795f049bbbf501276c7"; auth_token=f2f6a0fe18110ca7b47e841a60c2b7cb91f0f633; guest_id_ads=v1%3A170926443996790372; guest_id_marketing=v1%3A170926443996790372; guest_id=v1%3A170926443996790372; twid=u%3D1729666750357377024; ct0=9a1299de8acdccede72b5126031150df437091c8d69ec826e5dcd118126908217bc01584ae98a03c9a7b16d3c064d4b78e0f4b7e7faba8e6dd5dda95b54eb9193fa428b3596e3ba25aa73acf96ea90ad; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCESMGfiNAToMY3NyZl9p%250AZCIlZWRlNzg1MDE0NmU4NzJlZWRkOTJiNjYwYjU2NjgzOWM6B2lkIiU2ZDk1%250AZTk5MThjODEzNWUxNGZiNjc0NDJkMjg2YTZhYQ%253D%253D--d959c1a2159b05621afbadd321df636336e8688d; personalization_id="v1_W4zz9g42OiOoN62WXmvxig=="'
    ]

    records = pandas.read_excel(f'博文表.xlsx').to_dict(orient='records')
    current_blog = []
    count_current_blog = 0
    for itwitt in records:
        count_current_blog += 1
        flag = f'进度【{count_current_blog}/{len(records)}】'

        if itwitt['回复数量'] <= 0:
            continue

        tt = Gtwitter(10000)
        tt.blockdata(itwitt)
    pandas.DataFrame(current_blog).to_excel(f'评论表.xlsx', index=False)
