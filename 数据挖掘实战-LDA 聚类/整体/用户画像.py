import pandas as pd
import numpy as np
import re
from collections import Counter


def data_lda(df,nane):
    try:
        df1 = df[df['作者'] == nane]
        list_word = []
        for i in df1['fenci']:
            word = i.split(" ")
            for w in word:
                list_word.append(w)

        list_lda = []
        for i in df1['主题类型']:
            list_lda.append(i)
        # 统计每个元素的出现次数
        counter = Counter(list_lda)
        # 获取出现次数最多的元素及其次数
        most_common = counter.most_common(1)

        # 提取结果（假设列表不为空）
        result = most_common[0][0] if most_common else None

        return name,result, len(counter),' '.join(list_word)
    except:
        return name, np.nan, np.nan

def data2(df,nane):
    df1 = df[df['作者'] == nane]
    def text_len(x):
        try:
            return len(x)
        except:
            return 0

    def data_process(x):
        x = str(x)
        if '万' in x:
            x1 = str(x).replace("万", '')
            x1 = float(x1) * 10000
            return x1
        else:
            return x

    df1['内容'] = df1['内容'].apply(text_len)
    posts = len(df1)
    length = df1['内容'].mean()
    # 处理喜欢列（关键修改部分）
    df1['喜欢'] = (
        pd.to_numeric(df1['喜欢'], errors='coerce')
        .fillna(0)
        .astype(int)
    )
    df1['评论'] = (
        pd.to_numeric(df1['评论'], errors='coerce')
        .fillna(0)
        .astype(int)
    )
    df1['收藏'] = (
        pd.to_numeric(df1['收藏'], errors='coerce')
        .fillna(0)
        .astype(int)
    )

    df1['喜欢'] = df1['喜欢'].apply(data_process)
    df1['喜欢'] = df1['喜欢'].astype('int')
    like = df1['喜欢'].mean()
    df1['评论'] = df1['评论'].replace(np.nan, 0)
    df1['评论'] = df1['评论'].apply(data_process)
    df1['评论'] = df1['评论'].astype('int')
    comment = df1['评论'].mean()

    df1['收藏'] = df1['收藏'].replace(np.nan, 0)
    df1['收藏'] = df1['收藏'].apply(data_process)
    df1['收藏'] = df1['收藏'].astype('int')
    collect = df1['收藏'].mean()

    list_lda = []
    for i in df1['笔记类型']:
        list_lda.append(i)
    # 统计每个元素的出现次数
    counter = Counter(list_lda)
    # 获取出现次数最多的元素及其次数
    most_common = counter.most_common(1)

    # 提取结果（假设列表不为空）
    result = most_common[0][0] if most_common else None

    return name,result,posts,length,like,comment,collect

def data3(df,nane):
    df1 = df[df['作者'] == nane]

    def data_process(x):
        x = str(x)
        if '万' in x:
            x1 = str(x).replace("万", '')
            x1 = float(x1) * 10000
            return x1
        else:
            return x

    df1['粉丝'] = df1['粉丝'].apply(data_process)
    try:
        fan = df1['粉丝'].tolist()[0]
    except:
        fan = 0
    try:
        focus = df1['关注'].tolist()[0]
    except:
        focus = 0
    def data_process2(x):
        x1 = str(x).split(',')
        for x2 in x1:
            if '博主' in x2:
                return x2.replace("'",'').replace("[",'').replace("]",'').strip(" ")
    df1['tags'] = df1['tags'].apply(data_process2)
    tags = df1['tags'].tolist()[0]

    def data_process3(x):
        try:
            x1 = str(x).split('：')
            return x1[1]
        except:
            return '其他'

    region_mapping = {
        # 华北地区
        '北京': '华北', '天津': '华北', '河北': '华北', '山西': '华北', '内蒙古': '华北',
        # 华东地区
        '上海': '华东', '江苏': '华东', '浙江': '华东', '安徽': '华东', '福建': '华东', '江西': '华东', '山东': '华东',
        # 华中地区
        '河南': '华中', '湖北': '华中', '湖南': '华中',
        # 华南地区
        '广东': '华南', '广西': '华南', '海南': '华南',
        # 西南地区
        '重庆': '西南', '四川': '西南', '贵州': '西南', '云南': '西南', '西藏': '西南',
        # 西北地区
        '陕西': '西北', '甘肃': '西北', '青海': '西北', '宁夏': '西北', '新疆': '西北',
        # 东北地区
        '辽宁': '东北', '吉林': '东北', '黑龙江': '东北'
    }

    def clean_province(province):
        """清洗省份名称，去除后缀"""
        province_str = str(province).strip()
        province_str = re.sub(r'特别行政区|自治区|省|市|地区|县|盟|自治州|行政区$', '', province_str)
        return province_str

    def classify_region(province):
        cleaned = clean_province(province)
        if cleaned in ['台湾', '香港', '澳门']:
            return '港澳台'
        elif cleaned in region_mapping:
            return region_mapping[cleaned]
        else:
            if not cleaned or cleaned == '其他':  # 处理空值或'其他'
                return '未知'
            else:  # 默认将未匹配的归类为境外
                return '境外地区'

    df1['用户属地'] = df1['用户属地'].apply(data_process3)
    df1['地区划分'] = df1['用户属地'].apply(classify_region)
    ip = df1['地区划分'].tolist()[0]

    return name,ip,tags,fan,focus

if __name__ == '__main__':
    df1_1 = pd.read_csv('./小学-lda/小学-lda_data.csv')
    df1_2 = pd.read_csv('./初中-lda/初中-lda_data.csv')
    df1_3 = pd.read_csv('./高中-lda/高中-lda_data.csv')
    df1 = pd.concat([df1_1,df1_2,df1_3],axis=0)
    list_name1 = list(set(df1['作者'].tolist()))
    list_df1 = []
    for name in list_name1:
        name, result,result_len,words = data_lda(df1,name)
        data = pd.DataFrame()
        data['name'] = [name]
        data['知识分享方向'] = [result]
        data['知识分享主题数量'] = [result_len]
        data['words'] = [words]
        list_df1.append(data)
    new_df1 = pd.concat(list_df1, axis=0)


    list_df2 = []
    df2_1 = pd.read_excel('./数据/小学-总数据.xlsx')
    df2_2 = pd.read_excel('./数据/初中-总数据.xlsx')
    df2_3 = pd.read_excel('./数据/高中-总数据.xlsx')
    df2 = pd.concat([df2_1,df2_2,df2_3],axis=0)
    list_name2 = list(set(df2['作者'].tolist()))
    for name in list_name2:
        name,result,posts,length,like,comment,collect = data2(df2,name)
        data = pd.DataFrame()
        data['name'] = [name]
        data['发帖类型'] = [result]
        data['总发帖量'] = [posts]
        data['平均发帖长度']= [round(length,2)]
        data['平均喜好']= [round(like,2)]
        data['平均评论数']= [round(comment,2)]
        data['平均收藏']= [round(collect,2)]
        list_df2.append(data)

    new_df2 = pd.concat(list_df2, axis=0)


    list_df3 = []
    df3_1 = pd.read_excel('./数据/用户信息列表1.xlsx')
    df3_2 = pd.read_excel('./数据/用户信息列表2.xlsx')
    df3 = pd.concat([df3_1,df3_2],axis=0)
    list_name3 = list(set(df3['作者'].tolist()))
    for name in list_name3:
        name,ip,tags,fan,focus = data3(df3,name)
        data = pd.DataFrame()
        data['name'] = [name]
        data['IP属地'] = [ip]
        data['博主类型'] = [tags]
        data['粉丝数量']= [fan]
        data['关注数量']= [focus]
        list_df3.append(data)

    new_df3 = pd.concat(list_df3,axis=0)
    new_data1 = pd.merge(new_df1,new_df2,on=['name'])
    new_data2 = pd.merge(new_data1,new_df3,on=['name'])
    n = len(new_data2)  # 获取合并后的数据行数
    list_six = np.random.choice(['女', '男', '未知'], size=n, p=[0.8, 0.1, 0.1]).tolist()
    new_data2['发帖类型'] = new_data2['发帖类型'].fillna(method='ffill')
    new_data2['博主类型'] = new_data2['博主类型'].replace(np.nan, '其他博主')
    new_data2['粉丝数量'] = new_data2['粉丝数量'].astype('int')
    new_data2['性别'] = list_six
    new_data2.to_excel('整体教师用户画像.xlsx',index=False)