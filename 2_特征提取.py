import pandas as pd
import datetime
import os
import numpy as np
from collections import Counter
import math

#找一下角度差
def angle_dif(result):
    loc1=list(zip(result.longitude,result.latitude))
    loc2=list(zip(result.longitude_s,result.latitude_s))
    sum=list(zip(loc1,loc2))
    ang_df1=[]
    for i in sum:
        lat1,lon1=i[0]
        lat2,lon2=i[1]
        d=call_orientation(lat1, lon1,lat2, lon2)
        ang_df1.append(d)
    result['ang_df']=ang_df1
    result['wifi_count_rt']=result['wifishopcount'].astype('float') / result['shop_hot']
    result['top_count_rt'] = result['topcount'].astype('float')/result['shop_hot']
    result['rua8ratio']=result['rua8'].astype('float')/result['shop_hot']
    cols=['wifiprosum','knn_max_values','scorewifi','wifi_count_rt','knn_std']
    result,colsAfter=featureInGroup(result,cols)
    return result


def distance_dif(result):
    loc1=list(zip(result.longitude,result.latitude))
    loc2=list(zip(result.longitude_s,result.latitude_s))
    sum=list(zip(loc1,loc2))
    dis_df1=[]
    for i in sum:
        lat1,lon1=i[0]
        lat2,lon2=i[1]
        d=call_distance(lat1, lon1,lat2, lon2)
        dis_df1.append(d)
    result['dis_df']=dis_df1
    return result


def call_orientation(lat1,lon1,lat2,lon2):
    radLat1 = math.radians(lat1)
    radLon1 = math.radians(lon1)
    radLat2 = math.radians(lat2)
    radLon2 = math.radians(lon2)
    dLon = radLon2 - radLon1
    y = math.sin(dLon) * math.cos(radLat2)
    x = math.cos(radLat1) * math.sin(radLat2) - math.sin(radLat1) * math.cos(radLat2) * math.cos(dLon)
    brng = math.degrees(math.atan2(y, x))
    brng = (brng + 360) % 360
    return brng

def angle_real_dif(result):
    loc1=list(zip(result.longitude,result.latitude))
    loc2=list(zip(result.lon_real,result.lat_real))
    sum=list(zip(loc1,loc2))
    ang_df1=[]
    for i in sum:
        lat1,lon1=i[0]
        lat2,lon2=i[1]
        d=call_orientation(lat1, lon1,lat2, lon2)
        ang_df1.append(d)
    result['real_ang_df']=ang_df1
    return result


# 计算两点之间距离
def call_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L

def distance_real_dif(result):
    loc1=list(zip(result.longitude,result.latitude))
    loc2=list(zip(result.lon_real,result.lat_real))
    sum=list(zip(loc1,loc2))
    dis_df1=[]
    for i in sum:
        lat1,lon1=i[0]
        lat2,lon2=i[1]
        d=call_distance(lat1, lon1,lat2, lon2)
        dis_df1.append(d)
    result['real_dis_df']=dis_df1
    return result


#找一下距离差
#加上商店历史上的平均距离，商店位置表中的距离不准
def get_real_shop_loc(result):
    global shop_loc
    shop_loc=shop_loc.groupby('shop_id',as_index=False).agg({'longitude':'mean','latitude':'mean'})
    shop_loc.rename(columns={'longitude':'lon_real','latitude':'lat_real'},inplace=True)
    result=pd.merge(result,shop_loc,on='shop_id',how='left')
    shop_loc.rename(columns={'lon_real':'longitude','lat_real':'latitude'},inplace=True)
    return result


def get_time_hour(data):
    data['hourofday']=pd.DatetimeIndex(data.time_stamp).hour
    data['dayofweek']=pd.DatetimeIndex(data.time_stamp).day
    return data


#行为发生时间
def get_time(train,result):
    result=get_time_hour(result)
    train=get_time_hour(train)
    t1=train.groupby('shop_id',as_index=False)['hourofday'].agg({'h_s_m':'mean','h_s_v':'std','h_s':'median'})
    t2=train.groupby(['shop_id','hourofday'],as_index=False)['row_id'].agg({'shophourcount':'count'})
    result=pd.merge(result,t1,on='shop_id',how='left')
    result=pd.merge(result,t2,on=['shop_id','hourofday'],how='left')
    result['time_dif']=abs(result['h_s']-result['hourofday'])
    return result

#shop_id和当前信号最强的wifi以及第二强的wifi有没有互联记录
def if_wifi_shop_connected(result):
    global wifi_shop_connect_only
    wifi_shop_connect_only.rename(columns={'wifi_id':'wifi1_id','connect':'connect1'},inplace=True)
    result=pd.merge(result,wifi_shop_connect_only,on=['wifi1_id','shop_id'],how='left')
    wifi_shop_connect_only.rename(columns={'wifi1_id':'wifi2_id','connect1':'connect2'},inplace=True)
    result=pd.merge(result,wifi_shop_connect_only,on=['wifi2_id','shop_id'],how='left')
    wifi_shop_connect_only.rename(columns={'wifi2_id':'wifi_id','connect2':'connect'},inplace=True)
    result['connect1']=result['connect1'].fillna(0)
    result['connect2']=result['connect2'].fillna(0)
    result['connect_sum']=result['connect1']+result['connect2']
    return result

'''
#28 Knn_values是当前wifi序列的能量与历史商店平均能量的方差
#29 knn_std是当前wifi序列的能量与历史商店平均能量的差值的标准差
#30 knn_mean是当前wifi序列的能量与历史商店平均能量的差值的均值
#31 avgcossimilarity是当前wifi序列的能量与历史商店平均能量序列的cos相似度
#32larger1是当前wifi序列中大于历史商店平均能量的数量
#33larger1_ratio是当前wifi序列中大于历史商店平均能量的数量占当前wifi序列与历史商店wifi序列相同wifi个数的比例
#34powersmallest是当前序列wifi存在在商店历史中的最小能量
#35short2是当前wifi序列中小于历史商店平均能量的数量
#36rua1是当前wifi序列的能量与历史商店平均能量的带有权重的差值，这个权重具体看我的函数
#37rua3是当前序列wifi的能量与历史商店出现频率大于0.5的wifi的平均能量的方差
#38rua5是在result中每个row_id出现次数最多的10个wifi的历史在商店中的平均能量与当前序列的能量差
#39rua6是在result中每个row_id出现次数最多的10个wifi的历史在商店中的wifi出现在当前序列且出现在商店历史中的个数
#40short1是是当前wifi序列中小于历史商店最小能量的数量
#41knn_min_values是当前wifi序列的能量与商店历史wifi序列的最小能量的方差
#42short3是是当前wifi序列中大于历史商店最小能量的数量
#43short3_ratio是当前wifi序列wifi序列中大于历史商店最小能量的数量占相同数量的比例
#44knn_max_values是当前wifi序列与历史商店最大能量的方差
#45 maxcossimilarity是当前wifi序列的能量与历史商店最大能量序列的cos相似度
#46rua1是当前wifi序列的能量与历史商店最大能量的带有权重的差值，这个权重具体看我的函数
#47larger2是当前wifi序列中大于历史商店最大能量的数量
#48rua4是当前序列wifi的能量与历史商店出现频率大于0.5的wifi的最大能量的方差
#49rua7是在result中每个row_id出现次数最多的10个wifi的历史在商店中的最大能量与当前序列的能量差
#50rua8是当前序列中wifi能量大于历史上这个商店出现过的所有商店的能量的次数
#51rua9是当前序列中wifi能量大于历史上这个商店出现过的所有商店的能量的次数除以当前序列wifi在历史上出现的次数
#52rua10是当前序列中的wifi能量与历史上这个wifi出现过的最大能量的距离
#53rua11是当前序列中的wifi能量与历史上这个wifi出现过的最大能量的平均距离
#54rua12是当前序列中的wifi能量与历史上这个wifi出现过的最小能量的平均距离
'''
# def ger_power_var(train, result):
#     result['w4'] = result['wifi_infos'].map(get_wifi_power)
#     wifi_shop_info['w'] = wifi_shop_info['wifi_id'] + ':' + wifi_shop_info['ave_power'].astype('str')
#     t1 = wifi_shop_info[['shop_id', 'w']]
#     t2 = t1.groupby('shop_id')['w'].agg(lambda x: '|'.join(x)).reset_index()
#     t2['w'] = t2['w'].map(get_shop_wifi_dic)
#     result = pd.merge(result, t2, on=['shop_id'], how='left')
#     result['knn_values'] = map(lambda x, y: get_knn(x, y), result['w4'], result['w'])
#     result['knn_std'] = map(lambda x, y: get_std(x, y), result['w4'], result['w'])
#     result['knn_mean'] = map(lambda x, y: get_mean(x, y), result['w4'], result['w'])
#     result['avgcossimilarity'] = map(lambda x, y: get_cos(x, y), result['w4'], result['w'])
#     result['larger1'] = map(lambda x, y: find_larger_avp(x, y), result['w4'], result['w'])
#     result['larger1_ratio'] = map(lambda x, y: find_bigger_ratio(x, y), result['w4'], result['w'])
#     result['powersmallest'] = map(lambda x, y: find_smallest(x, y), result['w4'], result['w'])
#     result['short2'] = map(lambda x, y: find_short_min(x, y), result['w4'], result['w'])
#     result['rua1'] = map(lambda x, y, z: get_weighted_knn(x, y, z), result['w4'], result['w'], result['s'])
#     result['rua3'] = map(lambda x, y, z: get_knn1(x, y, z), result['w4'], result['w'], result['p'])
#     result['rua5'] = map(lambda x, y, z: get_knn2(x, y, z), result['w4'], result['w'], result['wx'])
#     result['rua6'] = map(lambda x, y, z: get_sametopwifi(x, y, z), result['w4'], result['w'], result['wx'])
#
#     wifi_shop_min['n1'] = wifi_shop_min['wifi_id'] + ':' + wifi_shop_min['min_power'].astype('str')
#     t1 = wifi_shop_min[['shop_id', 'n1']]
#     t2 = t1.groupby('shop_id')['n1'].agg(lambda x: '|'.join(x)).reset_index()
#     t2['n1'] = t2['n1'].map(get_shop_wifi_dic)
#     result = pd.merge(result, t2, on=['shop_id'], how='left')
#     result['short1'] = map(lambda x, y: find_short_min(x, y), result['w4'], result['n1'])
#     result['knn_min_values'] = map(lambda x, y: get_knn(x, y), result['w4'], result['n1'])
#     result['short3'] = map(lambda x, y: find_bigger_max(x, y), result['w4'], result['n1'])
#     result['short3_ratio'] = map(lambda x, y: find_bigger_ratio(x, y), result['w4'], result['n1'])
#     wifi_shop_max['n'] = wifi_shop_max['wifi_id'] + ':' + wifi_shop_max['max_power'].astype('str')
#     t1 = wifi_shop_max[['shop_id', 'n']]
#     t2 = t1.groupby('shop_id')['n'].agg(lambda x: '|'.join(x)).reset_index()
#     t2['n'] = t2['n'].map(get_shop_wifi_dic)
#     result = pd.merge(result, t2, on=['shop_id'], how='left')
#     result['knn_max_values'] = map(lambda x, y: get_knn(x, y), result['w4'], result['n'])
#     result['maxcossimilarity'] = map(lambda x, y: get_cos(x, y), result['w4'], result['n'])
#     result['rua2'] = map(lambda x, y, z: get_weighted_knn(x, y, z), result['w4'], result['n'], result['s'])
#     result['larger2'] = map(lambda x, y: find_bigger_max(x, y), result['w4'], result['n'])
#     result['rua4'] = map(lambda x, y, z: get_knn1(x, y, z), result['w4'], result['n'], result['p'])
#     result['rua7'] = map(lambda x, y, z: get_knn2(x, y, z), result['w4'], result['n'], result['wx'])
#     t3 = train[['shop_id', 'wifi_infos', 'row_id']].drop_duplicates()
#     t3 = f_owen(t3)
#     t3['allwifipower'] = t3['wifi_id'] + ':' + t3['power'].astype('str')
#     t3 = t3[['shop_id', 'allwifipower']]
#     t4 = t3.groupby('shop_id')['allwifipower'].agg(lambda x: '|'.join(x)).reset_index()
#     t4['allwifipower'] = t4['allwifipower'].map(get_shop_wifi_alldic)
#     result = pd.merge(result, t4, on=['shop_id'], how='left')
#     result['rua8'] = map(lambda x, y: get_infive(x, y), result['w4'], result['allwifipower'])
#     result['rua9'] = map(lambda x, y: get_better(x, y), result['w4'], result['allwifipower'])
#     global tx
#     t5 = train[['shop_id', 'wifi_infos']].drop_duplicates()  # todo
#     t6 = make_wifi_shop_relation(t5)
#     t6 = t6.groupby('wifi_id', as_index=False)['power'].agg({'max_power': 'max'})
#     tx = dict(zip(t6.wifi_id, t6.max_power))
#     result['rua10'] = map(lambda x, y: get_wifidistance(x, y), result['w4'], result['n'])
#     result['rua11'] = result['w4'].map(get_max_wifi_dis)
#     global tx_min
#     t7 = make_wifi_shop_relation(t5)  # 加
#     t7 = t7.groupby('wifi_id', as_index=False)['power'].agg({'min_power': 'min'})  # 加
#     tx = dict(zip(t7.wifi_id, t7.min_power))
#     result['rua12'] = result['w4'].map(get_max_wifi_dismin)
#
#     del result['w']
#     del result['n']
#     del result['s']
#     del result['p']
#     del result['allwifipower']
#
#     return result

def wifishopmaxin(train, result):
    t1 = train[['shop_id', 'wifi_infos']].drop_duplicates()
    t2 = swp(t1)
    t5 = t2.copy()
    t2 = t2.groupby(['shop_id', 'wifi_id'], as_index=False)['power'].agg({'max_power': 'max'})
    t3 = t2.groupby('wifi_id', as_index=False)['max_power'].agg({'max_power': 'max'})
    t4 = pd.merge(t3, t2, on=['max_power', 'wifi_id'], how='left')
    t4['mostwifi'] = t4['wifi_id'] + ':' + t4['max_power'].astype('str')
    t4 = t4[['shop_id', 'mostwifi']]
    t4 = t4.groupby('shop_id')['mostwifi'].agg(lambda x: '|'.join(x)).reset_index()
    t4['mostwifi'] = t4['mostwifi'].map(get_shop_wifi_dic)
    result = pd.merge(result, t4, on=['shop_id'], how='left')
    result['ww'] = result['wifi_infos'].map(get_wifi_power)
    result['wifimostcount'] = map(lambda x, y: get_wifimostcount(x, y), result['ww'], result['mostwifi'])

    t5 = t5.groupby(['shop_id', 'wifi_id'], as_index=False)['power'].agg({'countall': 'count'})
    t6 = t5.groupby('shop_id', as_index=False)['countall'].agg({'countall': 'max'})
    t6 = pd.merge(t6, t5, on=['shop_id', 'countall'], how='left')
    t6 = t6.groupby('shop_id')['wifi_id'].agg(lambda x: '|'.join(x)).reset_index()
    t6['wifi_id'] = t6['wifi_id'].map(get_tolist)
    t6.rename(columns={'wifi_id': 'wifi_need'}, inplace=True)
    result = pd.merge(result, t6[['wifi_need', 'shop_id']], on=['shop_id'], how='left')
    result['mostcountin'] = map(lambda x, y: get_mostcountin(x, y), result['ww'], result['wifi_need'])

    t7 = t2.groupby(['shop_id'], as_index=False)['max_power'].agg({'max_power': 'max'})
    t8 = pd.merge(t7, t2, on=['max_power', 'shop_id'], how='left')
    t8 = t8.groupby('shop_id')['wifi_id'].agg(lambda x: '|'.join(x)).reset_index()
    t8['wifi_id'] = t8['wifi_id'].map(get_tolist)
    t8.rename(columns={'wifi_id': 'wifi1_need'}, inplace=True)
    result = pd.merge(result, t8, on=['shop_id'], how='left')
    result['mostpowerin'] = map(lambda x, y: get_mostcountin(x, y), result['ww'], result['wifi1_need'])
    del result['ww']
    return result


def choosesamewifi(x):
    b = []
    from collections import Counter
    f = sorted(Counter(x).items(),key=lambda x: x[1],reverse=True)
    for i in range(10):
        try:
           b.append(f[i][0])
        except:
           return b
    return b
def listsum(x):
    a=[]
    for i in x:
        a=a+i
    return a

def samewifi(result):
    t1=result[['row_id', 'w2']].groupby('row_id')['w2'].agg(listsum).reset_index()
    t1['wx'] = t1['w2'].map(choosesamewifi)
    result = pd.merge(result, t1[['row_id', 'wx']], on='row_id', how='left')
    del result['w2']
    return result

# # 匹配度
# def connectamount(result):
#     t1 = wifi_shop_connect_only[['shop_id','wifi_id']].groupby('shop_id')['wifi_id'].agg(lambda x:':'.join(x)).reset_index()
#     t1['h2']=t1['wifi_id'].map(fuck)
#     del t1['wifi_id']
#     result=pd.merge(result,t1,on=['shop_id'],how='left')
#     a=list(result['w1'])
#     b=list(result['h2'])
#     c=[]
#     for i in range(len(a)):
#         try:
#             d=len(set(a[i])&set(b[i]))
#             c.append(d)
#         except:
#             c.append(0)
#     result['h3']=c
#     del result['w1']
#     del result['h2']
#     return result

def get_wificount(x,y):
    count=0
    try:
       for i in x:
           if i in y:
              count=count+int(x[i])
           else:
               count=count+0
       return count
    except:
       return 0

def get_shop_count_dic(x):
    wifi=[]
    count=[]
    wifis=x.split('|')
    for i in wifis:
        s=i.split(':')
        wifi.append(s[0])
        count.append(int(s[1]))
    wificount = dict(zip(wifi, count))
    return wificount

# wifishopcount是统计当前序列的wifi在历史上的次数
def wifi_count_intest(result):
    wifi_shop_count = get_wifi_shop_count(train)
    wifi_shop_count['w8'] = wifi_shop_count['wifi_id'] + ':' + wifi_shop_count['wifi inshopcount'].astype('str')
    t1 = wifi_shop_count[['shop_id','w8']]
    t2 = t1.groupby('shop_id')['w8'].agg(lambda x:'|'.join(x)).reset_index()
    t2['w8'] = t2['w8'].map(get_shop_count_dic)
    del wifi_shop_count['w8']
    result = pd.merge(result, t2, on = ['shop_id'],how='left')
    result['wifishopcount'] = map(lambda x, y: get_wificount(x, y) , result['w8'], result['w1'])
    del result['w8']
    return result

#组内比例
def _featureInGroupSingle(df, col):
    nameSum = '_sum'
    nameRate = '_groupRate'
    dfx = df.groupby('row_id', as_index=False)[col].agg({col+nameSum: 'sum'})
    dfx = pd.merge(df, dfx, how='left', on='row_id')
    dfx[col+nameRate] = dfx[col]/dfx[col+nameSum]
    return dfx[['row_id', col+nameRate]]

#注：组内Rate可以，组内Rank无效
def featureInGroup(df,cols):
    afterCol = []
    nameRate = '_groupRate'
    for col in cols:
        #print('>>>>>',col)
        dfy = _featureInGroupSingle(df[['row_id', col]], col)
        df[col+nameRate] = dfy[col+nameRate]
        afterCol.append(col+nameRate)
    return df, afterCol

# 负5.2 ['row_id', 'wifi_id', 'power']
def rwp(test):
    fuck = test.values
    res1 = []
    res2 = []
    res3 = []
    for i in fuck:
        wifis = i[1].split(';')
        for j in wifis:
            wifi = j.split('|')
            res1.append(i[0])
            res2.append(wifi[0])
            res3.append(int(wifi[1]))
    t_data = pd.DataFrame()
    t_data['row_id'] = res1
    t_data['wifi_id'] = res2
    t_data['power'] = res3
    return t_data


def featureLCS(train, test, result):
    t1 = train[['shop_id', 'wifi_infos']].drop_duplicates()
    t1 = swp(t1)
    t2 = test[['row_id', 'wifi_infos']].drop_duplicates()
    t2 = rwp(t2)
    dfx = t1.groupby(['shop_id', 'wifi_id'], as_index=False).power.agg({'bsCount': 'count', 'bsMedian': 'median'})
    dfx['srx'] = dfx.groupby('shop_id').bsCount.rank(ascending=False, method='min')
    dfx.srx = -dfx.srx
    # 归一化，平衡shop的交易量差异问题
    dft = dfx.groupby(['shop_id'], as_index=False).bsCount.agg({'bsMax': 'max', 'bsMin': 'min'})
    dfx = pd.merge(dfx, dft, how='left', on='shop_id')
    dft = dfx.groupby(['shop_id'], as_index=False).srx.agg({'srxMax': 'max', 'srxMin': 'min'})
    dfx = pd.merge(dfx, dft, how='left', on='shop_id')
    dfx['bsRate'] = (dfx.bsCount - dfx.bsMin) / (dfx.bsMax - dfx.bsMin)
    dfx['srxRate'] = (dfx.srx - dfx.srxMin) / (dfx.srxMax - dfx.srxMin)
    # 删除无用行
    dfx = dfx.drop(['srx', 'bsMax', 'bsMin', 'srxMax', 'srxMin', 'bsCount'], axis=1)
    # 记录序列————wifi强度和强度的Rank
    # 去重
    dfy = t2.groupby(['row_id', 'wifi_id'], as_index=False).head(1)
    dfy['sry'] = dfy[['row_id', 'wifi_id', 'power']].groupby('row_id').power.rank(ascending=False, method='min')
    dfy.sry = -dfy.sry
    # 归一化，平衡记录内部
    dft = dfy.groupby(['row_id'], as_index=False).sry.agg({'sryMax': 'max', 'sryMin': 'min'})
    dfy = pd.merge(dfy, dft, how='left', on='row_id')
    dfy['sryRate'] = (dfy.sry - dfy.sryMin) / (dfy.sryMax - dfy.sryMin)
    # 删除无用行
    dfy = dfy.drop(['sryMax', 'sryMin', 'sry'], axis=1)

    # 合并
    dfz = pd.merge(dfy, dfx, how='left', on='wifi_id')
    del dfx, dfy
    # 如果数据量太大，在此处筛选一波dfz
    # 强度差值及归一化，平衡不同bssid-shopid组合强度差的差异
    dfz['ds'] = -abs(dfz.power - dfz.bsMedian)
    dft = dfz.groupby(['wifi_id', 'shop_id'], as_index=False).ds.agg({'dsMax': 'max', 'dsMin': 'min'})
    dfz = pd.merge(dfz, dft, how='left', on=['wifi_id', 'shop_id'])
    dfz['dsRate'] = (dfz.ds - dfz.dsMin) / (dfz.dsMax - dfz.dsMin)
    # 赋权融合
    i, j, m, n = (0.3, 0.3, 0, 0.6)
    dfz['sr'] = i * dfz.srxRate + j * dfz.bsRate + m * dfz.sryRate + n * dfz.dsRate
    dfz = dfz.drop(['dsMax', 'dsMin'], axis=1)
    # 同类求和
    dfz = dfz.groupby(['row_id', 'shop_id'], as_index=False).sum()
    result = pd.merge(result, dfz[['row_id', 'shop_id', 'sryRate', 'bsRate', 'srxRate', 'dsRate', 'ds', 'sr']],
                      how='left', on=['row_id', 'shop_id'])
    cols = ['sryRate', 'bsRate', 'srxRate', 'dsRate', 'ds', 'sr']
    result, colsAfter = featureInGroup(result, cols)
    return result


# result['shop_id'], result['shop_score']
def most_frequency(x,y):
    i = Counter(y).most_common(20)
    try:
        return i[x]
    except:
        return 0

# w1
def w_find_s(x):
    d = []
    for i in x:
        # b_55477572': {'s_83428'}, 'b_55478310': {'s_267402'}, 'b_55478430': {'s_487166', 's_1813979'}
        if i in shop_wifi:
           for j in shop_wifi[i]:
               d.append(j)
    return d

def get_shop_score(train,result):
    global shop_wifi
    result['w1'] = result['wifi_infos'].map(wifi_name)  # w1是一个wifi列表
    wifi_shop_count = get_wifi_shop_count(train)
    t1= train[['shop_id','wifi_infos']].drop_duplicates()
    t1=swp(t1)
    t1=t1.groupby('wifi_id',as_index=False)['shop_id'].agg({'wifi_hot':'count'})
    t2=pd.merge(wifi_shop_count,t1,on=['wifi_id'],how='left')
    t2['pro']=t2['wifi inshopcount'].astype('float')/t2['wifi_hot']
    t2.sort_values(['wifi_id','pro'],inplace=True)
    t2=t2.groupby('wifi_id').tail(10)
    m=t2[['shop_id','wifi_id']].values
    # 'b_55477572': {'s_83428'}, 'b_55478310': {'s_267402'}, 'b_55478430': {'s_487166', 's_1813979'}
    shop_wifi = dict()
    for i in m:
       if i[1] not in shop_wifi:
          shop_wifi[i[1]] = set()
       shop_wifi[i[1]].add(i[0])
    # [s_2109, s_48050, s_2859627, s_405224, s_29667]
    result['shop_score'] = result['w1'].map(w_find_s)
    result['scorewifi'] = list(map(lambda x, y: most_frequency(x, y), result['shop_id'], result['shop_score']))
    print(set(result['scorewifi']))
    del result['shop_score']
    return result


def get_wifimulti(x,y):
    count=1
    try:
        for i in x:
            if i in y:
               count=count*(1-y[i])
            else:
                count = count
        return count
    except:
        return 1

def shop_pro(train,result):
   t1 = train[['shop_id','wifi_infos']].drop_duplicates()
   t1 = swp(t1)
   t1 = t1.groupby('wifi_id', as_index=False)['shop_id'].agg({'wifi_hot': 'count'})
   wifi_shop_count = get_wifi_shop_count(train)
   t2 = pd.merge(wifi_shop_count, t1, on=['wifi_id'], how='left')

   t2['pro'] = t2['wifi inshopcount'].astype('float') / t2['wifi_hot']
   t2['s'] = t2['wifi_id'] + ':' + t2['pro'].astype('str')
   t2 = t2[['shop_id', 's']]
   t2 = t2.groupby('shop_id')['s'].agg(lambda x:'|'.join(x)).reset_index()
   t2['s'] = t2['s'].map(f_s)
   result = pd.merge(result, t2, on=['shop_id'],how='left')
   result['wifiprosum'] = map(lambda x, y: get_wifipro(x, y) , result['w1'], result['s'])
   result['wifipromulti']=map(lambda x, y: get_wifimulti(x, y) , result['w1'], result['s'])
   #del result['s']
   return result


# result['w1'], result['p']
def get_wifipro(x, y):
    count = 0
    try:
        for i in x:
            if i in y:
                count = count + float(y[i])
            else:
                count = count + 0
        return count
    except:
        return 0

# result['w1'], result['p']
def get_wifinoshow(x, y):
    count = 0
    try:
        for i in x:
            if i in y:
                count = count + 1
            else:
                count = count + 0
        if len(y) > 10:
            return 10 - count
        else:
            return len(y) - count
    except:
        return 10


def get_wifiinrat(x, y):
    count = 0
    try:
        for i in x:
            if i in y:
                count = count + 1
            else:
                count = count + 0
        if len(y) > 10:
            return float(count) / 10
        else:
            return float(count) / len(y)
    except:
        return 0


def get_same(x, y):
    count = 0
    try:
        for i in x:
            if i in y:
                count = count + 1
            else:
                count = count + 0
    except:
        return 0

def f_s(x):
    wifis = x.split('|')
    wifi = list(map(lambda x: x.split(':')[0], wifis))
    count = list(map(lambda x: float(x.split(':')[1]), wifis))
    wificount = dict(zip(wifi, count))
    return wificount


def get_top10_sameresult(train, result):
    result['w1'] = result['wifi_infos'].map(wifi_name)  # w1是一个wifi列表
    wifi_shop_count = get_wifi_shop_count(train)
    # hop_hot是训练集中某商店的热度，通过shop_id来并起来
    t1 = train.groupby('shop_id', as_index=False)['row_id'].agg({'shop_hot': 'count'})
    t2 = pd.merge(wifi_shop_count, t1, on='shop_id', how='left')
    # todo wifiratio是用历史上wifi_id，shop_id的统计次数除以历史上商店的热度 可以理解成是当前shop,出现当前wifi的概率
    t2['wifiratio'] = t2['wifi inshopcount'] / (t2['shop_hot'].astype(float))
    t3 = train[['shop_id', 'wifi_infos']].drop_duplicates()
    t3 = swp(t3)
    t3 = t3.groupby('wifi_id', as_index=False)['shop_id'].agg({'wifi_hot': 'count'})
    t2 = pd.merge(t2, t3, on='wifi_id', how='left')
    # todo wifi2ratio是用历史上wifi_id,shop-id的统计次数除以历史上wifi_id的热度，可以理解成当前成已知当前wifi,是这个shop的概率
    t2['wifi2ratio'] = t2['wifi inshopcount'] / (t2['wifi_hot'].astype(float))
    # 将大于wifiratio大于0.5的理解成应该出现的
    t4 = t2.copy()
    t4 = t4[t4['wifiratio'] >= 0.5]
    t4['p'] = t4['wifi_id'] + ':' + t4['wifi2ratio'].astype('str')
    t4 = t4[['shop_id', 'p']]
    # todo 语法：根据shopid，加入了所有的b_13587831:1.0，以|为间隔
    t4 = t4.groupby('shop_id')['p'].agg(lambda x: '|'.join(x)).reset_index()
    # p {'b_24734860': 0.5384615384615384, 'b_19961439... }
    t4['p'] = t4['p'].map(f_s)
    result = pd.merge(result, t4, on=['shop_id'], how='left')
    # w1 [b_8086987, b_38106941, b_41212455, b_8086986,...  ]
    # wifinoshow理解成应该出现却没出现的次数
    result['wifinoshow'] = list(map(lambda x, y: get_wifinoshow(x, y), result['w1'], result['p']))
    # wifiinrat理解成应该出现的次数，除以当前wifi序列与历史wifi序列概率大于0.5的序列相同的次数
    result['wifiinrat'] = list(map(lambda x, y: get_wifiinrat(x, y), result['w1'], result['p']))
    # 9wifitopcount理解成当前WiFi序列与历史wifi序列概率大于0.5的序列相同的次数
    result['wifitopcount'] = list(map(lambda x, y: get_same(x, y), result['w1'], result['p']))
    # wifiturerat理解成wifiratio和wifi2ratio相乘的结果，然后将两序列相同的wifi的相乘概率相加
    t2['turerat'] = t2['wifi2ratio'] * t2['wifiratio']
    t2['v'] = t2['wifi_id'] + ':' + t2['turerat'].astype('str')
    t2 = t2[['shop_id', 'v']]
    t2 = t2.groupby('shop_id')['v'].agg(lambda x: '|'.join(x)).reset_index()
    t2['v'] = t2['v'].map(f_s)
    result = pd.merge(result, t2, on=['shop_id'], how='left')
    result['wifiturerat'] = list(map(lambda x, y: get_wifipro(x, y), result['w1'], result['v']))
    del result['v']
    del result['p']
    return result


def colon(x):
    x = list(x.split(':'))
    return x

def wifi_name(x):
    wifi = x.split(';')
    return list(map(lambda x: x.split('|')[0], wifi))

def get_wifi_shop_count(train):
    wifi_train = train[['shop_id', 'wifi_infos']].drop_duplicates()
    wifi_shop_count = swp(wifi_train)
    wifi_shop_count = wifi_shop_count.groupby(
        ['shop_id', 'wifi_id'], as_index=False)['power'].agg({'wifi inshopcount': 'count'})  # todo AB计数的快速方法，power是幌子
    wifi_shop_count.sort_values(['shop_id', 'wifi inshopcount'], inplace=True)  # todo 两列排序，正序
    wifi_shop_count = wifi_shop_count.groupby('shop_id').tail(50)  # todo Returns last n rows of each group
    return wifi_shop_count

# 计算匹配度
def get_same_relation(result, train):
    wifi_shop_count = get_wifi_shop_count(train)
    result['w1'] = result['wifi_infos'].map(wifi_name)  # w1是一个wifi列表
    # todo 语法：根据shopid，加入了所有的wifi_id，以：间隔
    shop_listwifi = wifi_shop_count.groupby('shop_id')['wifi_id'].agg(lambda x: ':'.join(x)).reset_index()
    shop_listwifi['w2'] = shop_listwifi['wifi_id'].map(colon)  # w1也是一个wifi列表
    del shop_listwifi['wifi_id']
    result = pd.merge(result, shop_listwifi,on=['shop_id'],how='left')
    # todo 匹配度算法
    a = list(result['w1'])
    b = list(result['w2'])
    c = []
    for i in range(len(a)):
        try:
            d = len(set(a[i]) & set(b[i]))
            c.append(d)
        except:
            c.append(0)
    result['w3'] = c
    return result


def swrp_1(x):
    return x.split('|')[0]
def swrp_2(x):
    return x.split('|')[1]
def swrp(train): # 原f_owen, make_wifi_rowid_relation
    train = train.drop('wifi_infos', axis=1).join(train['wifi_infos'].str.split(';', expand=True).
                                   stack().reset_index(level=1, drop=True).rename('wifi'))
    train['wifi_id'] = train['wifi'].map(lambda x: swrp_1(x))
    train['power'] = train['wifi'].map(lambda x: swrp_2(x))
    del train['wifi']
    return train

def swp(train):
    train = train.drop('wifi_infos', axis=1).join(train['wifi_infos'].str.split(';', expand=True).
                                   stack().reset_index(level=1, drop=True).rename('wifi'))
    train['wifi_id'] = train['wifi'].map(lambda x: swrp_1(x))
    train['power'] = list(map(int, train['wifi'].map(lambda x: swrp_2(x))))
    return train[['shop_id', 'wifi_id', 'power']]

# 商店的各个wifi出现的平均能量
def get_wifi_shop_info(train):
    wifi_train = train[['shop_id', 'wifi_infos']].drop_duplicates()
    wifi_shop_info = swp(wifi_train)
    wifi_shop_info = wifi_shop_info.groupby(['shop_id', 'wifi_id'], as_index=False)['power'].agg({'ave_power': 'mean'})
    return wifi_shop_info

def power2(x):
    wifis = x.split(';')
    wifi = list(map(lambda x: x.split('|')[0], wifis))
    power = list(map(int, list(map(lambda x: x.split('|')[1], wifis))))
    wifipower = dict(zip(wifi, power))
    return sorted(wifipower.items(), key=lambda x: x[1], reverse=True)[1] if len(wifipower) > 1 else 0

def power1(x):
    wifis = x.split(';')
    wifi = list(map(lambda x: x.split('|')[0], wifis))
    power = list(map(int, list(map(lambda x: x.split('|')[1], wifis))))
    wifipower = dict(zip(wifi, power))
    return sorted(wifipower.items(), key=lambda x: x[1], reverse=True)[0]

# 分为两部分,强id，强power 及其平均power和差值
def get_wifi(result, train):
    result['wifi1'] = result['wifi_infos'].map(power1)  # map和函数的用法
    result['wifi2'] = result['wifi_infos'].map(power2)
    result['wifi1_id'] = result['wifi1'].map(lambda x: x[0])
    result['wifi1_power'] = result['wifi1'].map(lambda x: x[1])
    result['wifi2_id'] = result['wifi2'].map(lambda x: 0 if x == 0 else x[0])
    result['wifi2_power'] = result['wifi2'].map(lambda x: 0 if x == 0 else x[1])

    # wifi1_id及其topcount，特征就是统计某商店最强wifi的出现的次数
    swr = train[['shop_id', 'wifi_infos', 'row_id']].drop_duplicates()
    srwp = swrp(swr)
    # 每条交易记录最强的power
    rsp_max = srwp[['shop_id', 'row_id', 'power']].groupby(['row_id', 'shop_id'], as_index=False)['power'].agg({'power':max})
    rspw_max = pd.merge(rsp_max, srwp, on=['shop_id', 'row_id', 'power'], how='left')  # ['row_id', 'shop_id', 'power', 'wifi_id']
    w1_count = rspw_max[['shop_id', 'row_id', 'wifi_id']].groupby(['wifi_id', 'shop_id'], as_index=False)['row_id'].agg(
        {'topcount': 'count'})
    w1_count.rename(columns={'wifi_id': 'wifi1_id'}, inplace=True)
    result = pd.merge(result, w1_count, on=['shop_id', 'wifi1_id'], how='left')

    # ave_power1是历史上这个最大强度wifi_id，shop_id的平均强度
    shop_wifi_pmean = get_wifi_shop_info(train)
    shop_wifi_pmean.rename(columns={'wifi_id': 'wifi1_id'}, inplace=True)
    result = pd.merge(result, shop_wifi_pmean, on=['shop_id', 'wifi1_id'], how='left')
    result.rename(columns={'ave_power': 'ave_power1'}, inplace=True)  # 有空值
    ave_min = result['ave_power1'].min()
    result['ave_power1'] = result['ave_power1'].fillna(ave_min)
    # power_dif1是ave_power1和wifi1_power的差, power_dif1是ave_power1和wifi1_power的差
    result['power_dif1'] = abs((result['ave_power1'] - result['wifi1_power']).astype(np.float))
    # 重复上述动作
    shop_wifi_pmean.rename(columns={'wifi1_id': 'wifi2_id'}, inplace=True)
    result = pd.merge(result, shop_wifi_pmean, on=['shop_id', 'wifi2_id'], how='left')
    result.rename(columns={'ave_power': 'ave_power2'}, inplace=True)
    ave_min = result['ave_power2'].min()
    result['ave_power2'] = result['ave_power2'].fillna(ave_min)
    result['power_dif2'] = abs((result['ave_power2'] - result['wifi2_power']).astype(np.float))
    return result


# 用户爱好
def get_uesr_hobby(train, result):
    user_kind_count = train.groupby(['user_id', 'category_id'], as_index=False)['category_id'].agg(
        {'user_kind_count': 'count'})
    result = pd.merge(result, user_kind_count, on=['user_id', 'category_id'], how='left')
    return result

def get_shop_category(result):
    result = pd.merge(result, shop, on=['shop_id'], how='left')
    result['category_id'] = result['category_id'].fillna(0)
    result['category'] = result['category_id'].map(lambda x: '0' if x == 0 else str(x)[2:]).astype('int')
    return result

# 获取用户到某商店的次数
def get_user_shop_count(train, result):
    # 又使用了shop_id
    user_shop_count = train.groupby(['user_id', 'shop_id'], as_index=False)['user_id'].agg({'user_shop_count': 'count'})
    result = pd.merge(result, user_shop_count, on=['user_id', 'shop_id'], how='left')
    return result

def get_shop_hot(train, result):
    shop_hot = train.groupby('shop_id', as_index=False)['row_id'].agg({'shop_hot': 'count'})
    result = pd.merge(result, shop_hot, on=['shop_id'], how='left')
    return result

# 获取用户历史行为次数
def get_user_count(train, result):
    # todo 简单计数，shop_id是幌子
    user_count = train.groupby('user_id', as_index=False)['shop_id'].agg({'user_count': 'count'})
    result = pd.merge(result, user_count, on=['user_id'], how='left')
    return result


def featureEngineering(train, result):
    # result = get_user_count(train, result)
    # result = get_shop_hot(train, result)
    # result = get_user_shop_count(train, result)
    # result = get_shop_category(result)
    # result = get_uesr_hobby(train, result)
    # result = get_wifi(result, train)
    # result = get_same_relation(result,train)
    # result = get_top10_sameresult(train, result)
    # result = shop_pro(train, result)
    # result = get_shop_score(train, result)
    # result = featureLCS(train, test, result)
    # result = wifi_count_intest(result)
    # # result = connectamount(result)
    # result = samewifi(result)
    # result = wifishopmaxin(train, result)
    # # result = ger_power_var(train, result)
    # result = if_wifi_shop_connected(result)
    # result = get_time(train,result)
    # result = get_real_shop_loc(result)
    # result = distance_real_dif(result)
    # result = angle_real_dif(result)
    # result = distance_dif(result)
    # result = angle_dif(result)
    # result['user_id']= result.user_id.map(lambda x:int(str(x)[2:]))
    # result['mall_id'] = result.mall_id.map(lambda x:int(str(x)[2:]))
    # result['real_lat_dif'] = abs(result['lat_real']-result['latitude'])
    # result['real_lon_dif'] = abs(result['lon_real']-result['longitude'])
    # result.fillna(0, inplace=True)
    return result

def load_data():
    # 加载数据
    train = open(r'训练数据-ccf_first_round_user_shop_behavior.csv')
    shop = open(r'训练数据-ccf_first_round_shop_info.csv')
    train = pd.read_csv(train)
    shop = pd.read_csv(shop)

    # # 使用交易发生的经纬度并更改shop的经纬度名字
    # shop_loc = train[['shop_id', 'longitude', 'latitude']]
    shop.rename(columns={'longitude': 'longitude_s', 'latitude': 'latitude_s'}, inplace=True)

    # 每一条交易记录都增加了row_id, t1138014
    train['row_id'] = range(train.shape[0])
    train['row_id'] = 't' + train['row_id'].astype('str')

    # shop_mall表
    shop_mall = shop[['shop_id', 'mall_id']]

    # 分为3部分，train1(18),train2(7),test(7)
    train['day'] = train['time_stamp'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M').day)
    train1 = train[(train['day'] < 18)]
    train2 = train[(train['day'] >= 18) & (train['day'] < 25)]
    test = train[(train['day'] >= 25)]
    del train['day'], train1['day'], train2['day'], test['day']

    # 新train(25，包含了train2), train2(模拟test)， test,删去train2和test的shop_id
    train = pd.concat([train1, train2])
    train2 = pd.merge(train2, shop_mall, on=['shop_id'], how='left')
    test = pd.merge(test, shop_mall, on=['shop_id'], how='left')
    # real = test[['row_id', 'shop_id']] # 训练时用
    train = pd.merge(train, shop, on=['shop_id'], how='left')
    train1 = pd.merge(train1, shop, on=['shop_id'], how='left')
    del train2['shop_id'], test['shop_id']
    return train1, train2, train, test, shop


if __name__ == '__main__':
    train1, train2, train, test, shop = load_data()
    # # 不要重复提取特征：
    # path1 = 'trainset.csv'
    # path2 = 'testset.csv'
    # if os.path.exists(path1) & os.path.exists(path2):
    #     train_set = pd.read_csv(path1)
    #     train_set = featureEngineering(train1, train_set)
    #     test_set = pd.read_csv(path2)
    #     test_set = featureEngineering(train, test_set)
    # else:
    train_sample = pd.read_csv(r'sampel_train1.csv')
    train_set = featureEngineering(train1, train_sample)
    test_sample = pd.read_csv(r'sampel_test1.csv')
    test_set = featureEngineering(train, test_sample)
    # print(train_set)
    # train_set.to_csv('trainset.csv', index=False)
    # test_set.to_csv('testset.csv', index=False)
