import hashlib
import json
import os.path
import random
import re
import time

import pandas
import requests
import ctypes
import struct

from lxml import etree

headers = {
    "user-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    "accept": 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    "cookie": '_zap=4b881bc2-be5f-439c-acfd-e08dcb522b2a; d_c0=ANBX8md15xaPTnkvoCsPlCtJ9vloDLd1eNU=|1686270046; YD00517437729195%3AWM_TID=l61xWLg5LKFAEBARRBaUkY4zNIMXkRSS; _xsrf=mgRpOxaoedCJ64yDCPxefCBf3CRWtso4; __snaker__id=ZtT1yUQ6HqGfFxBU; q_c1=d3a3cfad845e4a8c81a13f43af920f10|1689658901000|1689658901000; YD00517437729195%3AWM_NI=IBlaG9bbJ2H1Swh1ootuwN0FlRWQ37RGZnoVlEazw%2Bvic6hLtxqS7P1K0O6JTkU9yNcRXRBerscd7VALT5KsdlQNFoDhof9CVpIv7DMvOjyeUBL9L5eEJ%2BSdpboFGdCLcEw%3D; YD00517437729195%3AWM_NIKE=9ca17ae2e6ffcda170e2e6eeb9dc4d91908d98bb44f1ac8bb7c55e828f9eb1d867bbaaf9a5e144f8b5fd8be42af0fea7c3b92a8b9ea482c633a2be86d3cb3b91a9a8d2ee65b8abb9afb34eb08e99b2b23d9299968fee5eb09baf9bf621b6b7add7eb48b8a8bfb4b65cf39500b0b54298b59bdafc46a18e98b3c16fb2adad85b65e83949cd6ca79a8b49fafc56588b8a187bc4996b08797b6508faa818bfc6f979e84dacf3fa79b86ccd270b4bea1a7c6708194ada6f637e2a3; tst=r; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1698682116,1698682229,1698682725,1698682784; gdxidpyhxdE=TBzYQXpTu5kNXXGJlCHiIeoXRXGJwvf4DGuAvnJYKlGyDbNL%2BwjajUj%2FS9sMvIUJbE3hyALId6nCI%2BVxrxxsPUt8IG2elLZkpbVpVqAM9d3X%5CNX3r%2F%2FDP8TBSqGW0035gPzLSSSWxSxvEtPwDwovLcDOPk3%2FolQV%2BDUi0cyep5%2Fvusxb%3A1698688273343; captcha_ticket_v2={validate:CN31_cGZcsylB.42N6joi5fZei0Ue3jZ2sGxyFFOGez0OhqSpn8fnavTIRIeo*ULytWh*4t0IUFmSOLcfahFvMSZPrzUz..gdqkURDO3JM6WEFFYSNASx04EenzC5Lh6DY4qaR6R2hhTIoJGrqD5y9CvrJqBT6qTwAhJ9omWBOf32L6rR6*DzuBKVD*V6IRk5bOL9HtDXYYbwdnmbQK_K61t25LfUaRvqKhu0Pj5AwdeWqm3gdpc_aSXAul6xivYK0K4KcX8I8RfeeZcfhmot3_B4Gp_vK2i2yhJM.cRcLNyfF31zpJ04Su5Po.kJkBGVrCsLgBDWrptTcd_1w.Q3By4QeUf5c*4nrd21Sf24DKkEit*fQV*QqHVRCJRFfRAAUVfPEBPp6z3ibAcK5UuasxKggS1rSQtAwpk95IKvF*CNGjmzCeEJgmfhJrIRamPtU8L6jNHerghUAt3ebDxDWAtuxCKwQF9K06abqkyjhw8iIvD4DVgQo8ZkvjXbH2NkRIHvr5NliM77_v_i_1}; captcha_session_v2=2|1:0|10:1698687801|18:captcha_session_v2|88:YVB0cXRjZDB0N2hBOEkvWnROV2ZMV2c5anFzdENpMFhMRlBQcWwveXhvSmxaeEd3TjZNdENhWk8rYjFxMFdwdA==|d2abfe122e07d9aeaea014e221e0987d861eb729bdeb6ddf2957ce1753746ffd; o_act=login; ref_source=undefined; expire_in=15552000; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1698687818; z_c0=2|1:0|10:1698687818|4:z_c0|92:Mi4xQnlJbkJ3QUFBQUFBMEZmeVozWG5GaGNBQUFCZ0FsVk5Qemt0WmdCa3pKWTJGV1FHUWdjcElYVDhIYVFnaEl2LXB3|74f8596d1922df97f12cf4a5f050c63f89625c8beb84011c4e7aae1fe2e936a9; KLBRSID=d1f07ca9b929274b65d830a00cbd719a|1698687827|1698686863'
}


class x_zse_96_V3(object):
    local_48 = [48, 53, 57, 48, 53, 51, 102, 55, 100, 49, 53, 101, 48, 49, 100, 55]
    local_55 = "6fpLRqJO8M/c3jnYxFkUVC4ZIG12SiH=5v0mXDazWBTsuw7QetbKdoPyAl+hN9rgE"
    h = {
        "zk": [1170614578, 1024848638, 1413669199, -343334464, -766094290, -1373058082, -143119608, -297228157,
               1933479194, -971186181, -406453910, 460404854, -547427574, -1891326262, -1679095901, 2119585428,
               -2029270069, 2035090028, -1521520070, -5587175, -77751101, -2094365853, -1243052806, 1579901135,
               1321810770, 456816404, -1391643889, -229302305, 330002838, -788960546, 363569021, -1947871109],
        "zb": [20, 223, 245, 7, 248, 2, 194, 209, 87, 6, 227, 253, 240, 128, 222, 91, 237, 9, 125, 157, 230, 93, 252,
               205, 90, 79, 144, 199, 159, 197, 186, 167, 39, 37, 156, 198, 38, 42, 43, 168, 217, 153, 15, 103, 80, 189,
               71, 191, 97, 84, 247, 95, 36, 69, 14, 35, 12, 171, 28, 114, 178, 148, 86, 182, 32, 83, 158, 109, 22, 255,
               94, 238, 151, 85, 77, 124, 254, 18, 4, 26, 123, 176, 232, 193, 131, 172, 143, 142, 150, 30, 10, 146, 162,
               62, 224, 218, 196, 229, 1, 192, 213, 27, 110, 56, 231, 180, 138, 107, 242, 187, 54, 120, 19, 44, 117,
               228, 215, 203, 53, 239, 251, 127, 81, 11, 133, 96, 204, 132, 41, 115, 73, 55, 249, 147, 102, 48, 122,
               145, 106, 118, 74, 190, 29, 16, 174, 5, 177, 129, 63, 113, 99, 31, 161, 76, 246, 34, 211, 13, 60, 68,
               207, 160, 65, 111, 82, 165, 67, 169, 225, 57, 112, 244, 155, 51, 236, 200, 233, 58, 61, 47, 100, 137,
               185, 64, 17, 70, 234, 163, 219, 108, 170, 166, 59, 149, 52, 105, 24, 212, 78, 173, 45, 0, 116, 226, 119,
               136, 206, 135, 175, 195, 25, 92, 121, 208, 126, 139, 3, 75, 141, 21, 130, 98, 241, 40, 154, 66, 184, 49,
               181, 46, 243, 88, 101, 183, 8, 23, 72, 188, 104, 179, 210, 134, 250, 201, 164, 89, 216, 202, 220, 50,
               221, 152, 140, 33, 235, 214],
        "zm": [120, 50, 98, 101, 99, 98, 119, 100, 103, 107, 99, 119, 97, 99, 110, 111]
    }

    @staticmethod
    def pad(data_to_pad):
        padding_len = 16 - len(data_to_pad) % 16
        padding = chr(padding_len).encode() * padding_len
        return data_to_pad + padding

    @staticmethod
    def unpad(padded_data):
        padding_len = padded_data[-1]
        return padded_data[:-padding_len]

    @staticmethod
    def left_shift(x, y):
        x, y = ctypes.c_int32(x).value, y % 32
        return ctypes.c_int32(x << y).value

    @staticmethod
    def Unsigned_right_shift(x, y):
        x, y = ctypes.c_uint32(x).value, y % 32
        return ctypes.c_uint32(x >> y).value

    @classmethod
    def Q(cls, e, t):
        return cls.left_shift((4294967295 & e), t) | cls.Unsigned_right_shift(e, 32 - t)

    @classmethod
    def G(cls, e):
        t = list(struct.pack(">i", e))
        n = [cls.h['zb'][255 & t[0]], cls.h['zb'][255 & t[1]], cls.h['zb'][255 & t[2]], cls.h['zb'][255 & t[3]]]
        r = struct.unpack(">i", bytes(n))[0]
        return r ^ cls.Q(r, 2) ^ cls.Q(r, 10) ^ cls.Q(r, 18) ^ cls.Q(r, 24)

    @classmethod
    def g_r(cls, e):
        n = list(struct.unpack(">iiii", bytes(e)))
        [n.append(n[r] ^ cls.G(n[r + 1] ^ n[r + 2] ^ n[r + 3] ^ cls.h['zk'][r])) for r in range(32)]
        return list(
            struct.pack(">i", n[35]) + struct.pack(">i", n[34]) + struct.pack(">i", n[33]) + struct.pack(">i", n[32]))

    @classmethod
    def re_g_r(cls, e):
        n = [0] * 32 + list(struct.unpack(">iiii", bytes(e)))[::-1]
        for r in range(31, -1, -1):
            n[r] = cls.G(n[r + 1] ^ n[r + 2] ^ n[r + 3] ^ cls.h['zk'][r]) ^ n[r + 4]
        return list(
            struct.pack(">i", n[0]) + struct.pack(">i", n[1]) + struct.pack(">i", n[2]) + struct.pack(">i", n[3]))

    @classmethod
    def g_x(cls, e, t):
        n = []
        i = 0
        for _ in range(len(e), 0, -16):
            o = e[16 * i: 16 * (i + 1)]
            a = [o[c] ^ t[c] for c in range(16)]
            t = cls.g_r(a)
            n += t
            i += 1
        return n

    @classmethod
    def re_g_x(cls, e, t):
        n = []
        i = 0
        for _ in range(len(e), 0, -16):
            o = e[16 * i: 16 * (i + 1)]
            a = cls.re_g_r(o)
            t = [a[c] ^ t[c] for c in range(16)]
            n += t
            t = o
            i += 1
        return n

    @classmethod
    def b64encode(cls, md5_bytes: bytes, device: int = 0, seed: int = 63) -> str:
        local_50 = bytes([seed, device]) + md5_bytes  # 随机数  0 是环境检测通过
        local_50 = cls.pad(bytes(local_50))
        local_34 = local_50[:16]
        local_35 = [local_34[local_11] ^ cls.local_48[local_11] ^ 42 for local_11 in range(16)]
        local_36 = cls.g_r(local_35)
        local_38 = local_50[16:]
        local_39 = cls.g_x(local_38, local_36)
        local_53 = local_36 + local_39
        local_56 = 0
        local_57 = ""
        for local_13 in range(len(local_53) - 1, 0, -3):
            local_58 = 8 * (local_56 % 4)
            local_56 = local_56 + 1
            local_59 = local_53[local_13] ^ cls.Unsigned_right_shift(58, local_58) & 255
            local_58 = 8 * (local_56 % 4)
            local_56 = local_56 + 1
            local_59 = local_59 | (local_53[local_13 - 1] ^ cls.Unsigned_right_shift(58, local_58) & 255) << 8
            local_58 = 8 * (local_56 % 4)
            local_56 = local_56 + 1
            local_59 = local_59 | (local_53[local_13 - 2] ^ cls.Unsigned_right_shift(58, local_58) & 255) << 16
            local_57 = local_57 + cls.local_55[local_59 & 63]
            local_57 = local_57 + cls.local_55[cls.Unsigned_right_shift(local_59, 6) & 63]
            local_57 = local_57 + cls.local_55[cls.Unsigned_right_shift(local_59, 12) & 63]
            local_57 = local_57 + cls.local_55[cls.Unsigned_right_shift(local_59, 18) & 63]
        return local_57

    @classmethod
    def b64decode(cls, x_zse_96: str) -> dict:
        local_56 = 0
        local_57 = []
        for local_13 in range(0, len(x_zse_96), 4):
            local_59 = (cls.local_55.index(x_zse_96[local_13 + 3]) << 18) + (
                    cls.local_55.index(x_zse_96[local_13 + 2]) << 12) + (
                               cls.local_55.index(x_zse_96[local_13 + 1]) << 6) + cls.local_55.index(
                x_zse_96[local_13])
            local_58 = 8 * (local_56 % 4)
            local_56 = local_56 + 1
            local_57.append((local_59 & 255) ^ cls.Unsigned_right_shift(58, local_58))
            local_58 = 8 * (local_56 % 4)
            local_56 = local_56 + 1
            local_57.append(((local_59 >> 8) & 255) ^ cls.Unsigned_right_shift(58, local_58))
            local_58 = 8 * (local_56 % 4)
            local_56 = local_56 + 1
            local_57.append(((local_59 >> 16) & 255) ^ cls.Unsigned_right_shift(58, local_58))
        local_36, local_39 = local_57[-16:][::-1], local_57[:-16][::-1]
        local_38 = cls.re_g_x(local_39, local_36)
        local_35 = cls.re_g_r(local_36)
        local_34 = [local_35[local_11] ^ cls.local_48[local_11] ^ 42 for local_11 in range(16)]
        local_50 = cls.unpad(bytes(local_34 + local_38))
        return {
            'seed': local_50[0],
            'device': local_50[1],
            'md5_bytes': local_50[2:]
        }


def hash_md5(url):
    data = [
        "101_3_3.0",
        url,
        "ANBX8md15xaPTnkvoCsPlCtJ9vloDLd1eNU=|1686270046",
        "3_2.0aR_sn690QR2VMhnyT6Sm2JLBkhnun820XM20cL_1kwxYUqwT16P0EiUZcX2x-LOmwhp1tD_I-JOfgGXTzJO1ADRZ0cHsTJXII820Eer0c4nVDJH8zGCBADwMuukRe8tKIAtqS_L1VufXQ6P0mRPCyDQMovNYEgSCPRP0E4rZUrN9DDom3hnynAUMnAVPF_PhaueTF6CBhgV8eXpGZUO_FJNm3veseTCLnq3C-rUCuCHLobOMMUwCEgVYhCFs-9eTVXgOJ7pG_hLpJ6O8yqpyAgSMH9FmIJHqlcX0VJUCCCCOhUYGBJL1zqom6wNp2Lt1RwSG_qoBJbOsbJLKBbef0vxGeMtVkDHBg9xmIDc9NUosZu2_4q39Og_z9DHX8UNGDGY8TULBCUF90utGK6HfUbUCuCwsr7p04BoG64wmYqCBVvx12ckq1qH1JXCBngCmXBOMS0C1ywx8tUOOK8pLQvHYArx83CNGfwSfWJxC3rOs"
    ]
    return hashlib.md5('+'.join(data).encode()).hexdigest()


def getheaders(orgapi):
    signature = hash_md5(orgapi)
    x96 = x_zse_96_V3.b64encode(signature.encode(), seed=14)
    return {
        "x-zst-81": '3_2.0aR_sn690QR2VMhnyT6Sm2JLBkhnun820XM20cL_1kwxYUqwT16P0EiUZcX2x-LOmwhp1tD_I-JOfgGXTzJO1ADRZ0cHsTJXII820Eer0c4nVDJH8zGCBADwMuukRe8tKIAtqS_L1VufXQ6P0mRPCyDQMovNYEgSCPRP0E4rZUrN9DDom3hnynAUMnAVPF_PhaueTF6CBhgV8eXpGZUO_FJNm3veseTCLnq3C-rUCuCHLobOMMUwCEgVYhCFs-9eTVXgOJ7pG_hLpJ6O8yqpyAgSMH9FmIJHqlcX0VJUCCCCOhUYGBJL1zqom6wNp2Lt1RwSG_qoBJbOsbJLKBbef0vxGeMtVkDHBg9xmIDc9NUosZu2_4q39Og_z9DHX8UNGDGY8TULBCUF90utGK6HfUbUCuCwsr7p04BoG64wmYqCBVvx12ckq1qH1JXCBngCmXBOMS0C1ywx8tUOOK8pLQvHYArx83CNGfwSfWJxC3rOs',
        "x-zse-93": '101_3_3.0',
        "Accept": '*/*',
        "user-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        "x-zse-96": f'2.0_{x96}',
        "Referer": 'https://www.zhihu.com/topic/20675489/hot',
        "cookie": headers["cookie"]

    }



def send_req(firstUrl, headers):
    while 1:
        try:
            res = requests.get(
                firstUrl,
                headers=headers,
                timeout=(4, 5)
            )

            if res.status_code != 200:
                print(f"响应异常：请点击下方链接完成认证(半分后自动请求)!", res.json())
                time.sleep(30)
                continue
            return res.json()
        except Exception as e:
            print(f"error:{e}")
            time.sleep(1)




def getele(ele):
    try:
        return etree.HTML(ele).xpath("string(.)")
    except:
        return ele




def search_by_questionid(replyid, huatiname):
    next_page = {
        "page": f'/api/v4/comment_v5/answers/{replyid}/root_comment?order_by=score&limit=20&offset=',
    }
    for page in range(0, 1000):
        firstUrl = next_page["page"]
        print(f'firstUrl:{firstUrl}')
        headers = getheaders(firstUrl)
        nextapi = 'https://www.zhihu.com' + firstUrl
        print(f"firstUrl:{nextapi}")
        res = send_req(nextapi, headers)
        time.sleep(random.uniform(0, 1))
        print(res)
        for target in (res["data"]):
            try:
                saveitem = {}
                saveitem["电影名称"] = huatiname
                saveitem["问题id"] = replyid
                saveitem["评论id"] = target.get("id")

                saveitem["评论赞同数"] = target.get("like_count")
                saveitem["评论内容"] = getele(target.get("content"))
                saveitem["评论创建时间"] = time.strftime('%Y-%m-%d', time.localtime(target.get("created_time")))
                with open("comments.txt", 'a', encoding='utf-8') as f:
                    f.write(json.dumps(saveitem))
                    f.write('\n')
                print(huatiname, page, saveitem["评论内容"][:80] + "...")
            except Exception as e:
                print(f"parse error:{e}")
        if len(res["data"]) < 2 or res["paging"]["is_end"] is True:
            break
        else:
            next_page["page"] = res["paging"]["next"].split("www.zhihu.com")[-1]
            print(next_page["page"])

if __name__ == "__main__":
    records = pandas.read_excel("帖子.xlsx").to_dict(orient='records')

    for ire in records:
        if ire.get("回答评论数") <= 0:
            continue

        search_by_questionid(replyid=ire.get("回复id"), huatiname=ire.get("电影名称"))
    with open("comments.txt",'r',encoding='utf-8') as f:
        lines = [json.loads(i.strip()) for i in f.readlines()]
    df= pandas.DataFrame(lines)
    df.to_excel("帖子评论.xlsx",index=False)
