import time
import pandas as pd
import datetime


def get_label(data):
    # 真实交易记录对应的标签
    true = dict(zip(train['row_id'].values, train['shop_id']))
    # todo 可用的高级配对语法及比较语法
    data['label'] = data['row_id'].map(true)
    data['label'] = (data['label'] == data['shop_id']).astype('int')  # bool转化为0，1
    return data


# result3 通过k近邻及其geo权重推荐
def KnnGeo(train, test):
    from sklearn import neighbors
    gg = pd.DataFrame(columns=['row_id', 'shop_id', 'geoR'])
    train = pd.merge(train, shop_mall, on=['shop_id'], how='left')
    mallList = train.mall_id.unique()  # todo 不用set的去重语法
    shopIndex = list(train.columns).index('shop_id')  # 就是shop_id在表中的第几列
    n = 6  # knn的n
    # # todo 经纬度分商场
    for mall in mallList:
        tempTrain = train[train.mall_id == mall]
        tempTest = test[test.mall_id == mall]
        # 经纬度候选
        model = neighbors.KNeighborsClassifier(n_neighbors=n)
        model = model.fit(tempTrain[['longitude', 'latitude']].values, tempTrain['shop_id'].values)
        # todo kneighbors Finds the K-neighbors of a point,对于每个点，返回训练集中的n近邻点的下标
        neiborList = model.kneighbors(tempTest[['longitude', 'latitude']].values, return_distance=False)
        row_id_list = list(tempTest.row_id.values)
        gg_temp = pd.DataFrame(columns=['row_id', 'shop_id', 'geoR'])
        for i in range(n):
            gg_temp['row_id'] = row_id_list  # 添加n次row_id
            # 第一次添加最近邻的shop_id，第二次添加二近邻，以此类推
            gg_temp['shop_id'] = list(map(lambda x: tempTrain.iloc[x[i], shopIndex], neiborList))
            # 第一次为6，第二次为5，可能代表权重
            gg_temp['geoR'] = [n - i] * len(tempTest)
            gg = pd.concat([gg, gg_temp])
    result = gg.sort_values(['row_id', 'geoR'], ascending=False).groupby(['row_id', 'shop_id'], as_index=False).head(1)
    result = result[['row_id', 'shop_id']]
    return result


# result2
def user_shop(train, test):
    user_shop = train[['shop_id', 'user_id']].drop_duplicates()
    # todo 关键:都是以测试集为主表, 以user_id为主键是关键，之前一个user本就可以在多个商店中出现，通过它去推荐
    result = pd.merge(test[['user_id', 'row_id']], user_shop, on=['user_id'])  # 不用how，就没有空值
    result = result[['row_id', 'shop_id']].drop_duplicates()
    return result


def swrp_1(x):
    return x.split('|')[0]


def swrp_2(x):
    return x.split('|')[1]


def swrp(train):  # 原f_owen, make_wifi_rowid_relation
    train = train.drop('wifi_infos', axis=1).join(train['wifi_infos'].str.split(';', expand=True).
                                                  stack().reset_index(level=1, drop=True).rename('wifi'))
    train['wifi_id'] = train['wifi'].map(lambda x: swrp_1(x))
    train['power'] = train['wifi'].map(lambda x: swrp_2(x))
    del train['wifi']
    return train


# result1 通过wifi_id来构建样本
def wifi_shop(train, test):
    swr = train[['shop_id', 'wifi_infos', 'row_id']]  # 581726
    swpr = swrp(swr)  # ['shop_id', 'wifi_id', 'power', 'row_id']
    rwp = swrp(test)  # ['row_id', 'wifi_id', 'power']
    # 注：head(2)选出了前两个最大的（-46，-49） 商店的power
    dfx = swpr.sort_values(['row_id', 'power']).groupby(
        ['row_id'], as_index=False).head(2)  # ['shop_id', 'wifi_id', 'power', 'row_id']
    # 统计，wifi和shop个数，选出前5个,只能训练集 todo 二分类方法标签使用处
    dfx = dfx.groupby(['wifi_id', 'shop_id'], as_index=False).power.agg(
        {'bsCount': 'count'}).sort_values(['wifi_id', 'bsCount'])  # todo AB计数的快速方法，没有了power
    dfx = dfx.groupby(['wifi_id'], as_index=False).head(5)  # 此特征与负4对比 ['wifi_id', 'shop_id', 'bsCount']
    # print(dfx[dfx['wifi_id'] == 'b_9997654'])  # [wifi_id,user_id,time_stamp,longitude,latitude,row_id,mall_id,power]
    dfy = rwp.sort_values(['row_id', 'power']).groupby(['row_id'], as_index=False).head(3)
    # print(dfy[dfy['wifi_id'] == 'b_9997654'])
    # todo 关键:都是以测试集为主表, 以wifi_id为主键是关键，之前一个wifi本就可以在多个商店中出现，通过它去推荐
    dfz = pd.merge(dfx, dfy, how='right',
                   on='wifi_id')  # 右连接，以右表为主 ，['wifi_id', 'shop_id', 'bsCount', 'row_id', 'power']
    # print(dfz[dfz['wifi_id'] == 'b_9997654'])
    # 去重效果, 只留下head(1)
    dfz = dfz.sort_values(['row_id', 'shop_id', 'bsCount']).groupby(['row_id', 'shop_id'], as_index=False).head(1)
    dfz = dfz[['row_id', 'shop_id']]
    result = dfz.sort_values('row_id').reset_index(drop=True)
    result = result[(~result['shop_id'].isnull())]  # todo 语法：isnull取反 【row_id    shop_id】
    return result


# 构造样本
def make_sample(train, test):
    result1 = wifi_shop(train, test)  # 根据wifi历史来添加样本
    result2 = user_shop(train, test)  # 用户历史商铺推荐
    result3 = KnnGeo(train, test)  # 经纬度k近邻推荐
    # # result4 = GetLCSCandidate(train, test) # 废弃
    result = pd.concat([result1, result2, result3]).drop_duplicates()
    # todo 总的result(result为主表),row_id为主键
    result = pd.merge(result, test, on='row_id', how='left')
    del result['mall_id']
    return result


if __name__ == '__main__':
    t1 = time.clock()
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

    # 新train(25，包含了train2), train2(模拟test)，test        删去train2和test的shop_id
    train = pd.concat([train1, train2])
    train2 = pd.merge(train2, shop_mall, on=['shop_id'], how='left')
    test = pd.merge(test, shop_mall, on=['shop_id'], how='left')
    # real = test[['row_id', 'shop_id']] # 训练时用
    del train2['shop_id'], test['shop_id']

    print('make sample')

    train_data = make_sample(train1, train2)  # 构建候选集

    # 新的训练集
    train_data = get_label(train_data)  # 覆盖率0.945384

    train_data.to_csv('sampel_train1.csv', index=False)

    # 新的测试集，todo 没有getlabel
    test_data = make_sample(train, test)

    test_data.to_csv('sampel_test1.csv', index=False)

    print(time.clock() - t1)  # 用时约3min
