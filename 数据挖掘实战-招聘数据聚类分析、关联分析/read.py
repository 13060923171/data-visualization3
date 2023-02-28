import json

import pandas
import redis

redis_con = redis.Redis(db=3)



pandas.DataFrame([json.loads(i.decode()) for i in redis_con.hvals("jobdetail")]).to_excel("前程无忧-xlsx.xlsx")


pandas.DataFrame([json.loads(i.decode()) for i in redis_con.hvals("zhilian-detail")]).to_excel("智联-xlsx.xlsx")


pandas.DataFrame([json.loads(i.decode()) for i in redis_con.hvals("liepin-jobdetail")]).to_excel("猎聘-xlsx.xlsx")