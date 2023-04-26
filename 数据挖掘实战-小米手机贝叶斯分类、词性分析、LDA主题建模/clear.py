import pandas
import pymongo
import pymysql


def clear_shops():
    tables = database["shopdetail"]
    records = [i for i in tables.find({})]
    pandas.DataFrame(records).to_excel("产品信息.xlsx",index=False)


    mysql = pymysql.connect(
        user='root',
        password='961948438',
        database='xiaomi'
    )
    cursor = mysql.cursor()
    for irecord in records:
        listdatas = list(irecord.values())
        sql = "insert into shopdetail values('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" % (
            str(listdatas[0]),
            str(listdatas[1]),
            str(listdatas[2]),
            str(listdatas[3]),
            str(listdatas[4]),
            str(listdatas[5]),
            str(listdatas[6]),
            str(listdatas[7]),
            str(listdatas[8]),
            str(listdatas[9]),
            str(listdatas[10]),
            str(listdatas[11]),
            str(listdatas[12]),
            str(listdatas[13]),
            str(listdatas[14]),
            str(listdatas[15]),
            str(listdatas[16]),
            str(listdatas[17]),
            str(listdatas[18]),
            str(listdatas[19]),
        )
        try:
            cursor.execute(sql)
            mysql.commit()
            print(sql)
        except Exception as e:
            print(f"some error:{e}")




def clear_comments():
    tables = database["comment"]
    records = [i for i in tables.find({})]
    pandas.DataFrame(records).to_excel("评论信息.xlsx",index=False)


    mysql = pymysql.connect(
        user='root',
        password='961948438',
        database='xiaomi'
    )
    cursor = mysql.cursor()
    for irecord in records:
        listdatas = list(irecord.values())
        sql = "insert into comment values('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" % (
            str(listdatas[0]),
            str(listdatas[1]),
            str(listdatas[2]),
            str(listdatas[3]),
            str(listdatas[4]),
            str(listdatas[5]),
            str(listdatas[6]),
            str(listdatas[7]),
            str(listdatas[8]),
            str(listdatas[9]),
            str(listdatas[10]),
            str(listdatas[11]),
            str(listdatas[12]),
            str(listdatas[13]),
            str(listdatas[14]),
            str(listdatas[15]),
        )
        try:
            cursor.execute(sql)
            mysql.commit()
            print(sql)
        except Exception as e:
            print(f"some error:{e}")


if  __name__ == "__main__":

    mongodb_database = pymongo.MongoClient()
    database = mongodb_database["xm_database"]
    clear_shops()
    clear_comments()