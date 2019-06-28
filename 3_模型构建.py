import xgboost as xgb
import pandas as pd
import time
import datetime
from sklearn.metrics import accuracy_score

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
    real = test[['row_id', 'shop_id']]
    real.rename(columns={'shop_id': 'real'}, inplace=True)
    del train2['shop_id'], test['shop_id']
    return test, real





if __name__ == '__main__':
    t0 = time.clock()
    train_set = pd.read_csv(r'trainset.csv')
    test_set = pd.read_csv(r'testset.csv')
    test, real = load_data()
    print(train_set.columns)
    predictors = [x for x in train_set.columns if x not in ['row_id', 'shop_id', 'user_id', 'time_stamp', 'longitude', 'latitude', 'wifi_infos', 'label', 'wifi1', 'wifi2', 'category_id',
                                                            'mall_id', 'wifi1_id', 'wifi2_id']]
    print(predictors)
    print(train_set[predictors])

    params = {
        'objective': 'binary:logistic',
        'eta': 0.08,
        'colsample_bytree': 0.886,
        'min_child_weight': 1.1,
        'max_depth': 7,
        'subsample': 0.886,
        'gamma': 0.1,
        'lambda': 10,
        'verbose_eval': True,
        'eval_metric': 'auc',
        'scale_pos_weight': 6,
        'seed': 201703,
        'missing': -1
    }

    # label是标签
    xgbtrain = xgb.DMatrix(train_set[predictors], train_set['label'])
    xgbtest = xgb.DMatrix(test_set[predictors])
    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'test')]
    #
    print('start training')
    model = xgb.train(params, xgbtrain,  num_boost_round=500, evals=watchlist)
    # todo pred是概率
    test_set['pred'] = model.predict(xgbtest)
    test_pre = test_set[['row_id', 'shop_id', 'pred']].drop_duplicates()
    # 选出最大的pred,并且连回原表
    result = test_pre.groupby('row_id', as_index=False)['pred'].agg({'pred': 'max'})  # [row_id,pred]
    result = pd.merge(result, test_pre, on=['row_id', 'pred'], how='left')
    # 预测结果连接回测试集，没有预测结果的置0
    result = pd.merge(test[['row_id']], result[['row_id', 'shop_id']], on='row_id', how='left')
    result.fillna('0', inplace=True)
    score = pd.merge(real, result, on='row_id', how='left')
    print(accuracy_score(score['real'], score['shop_id']))
    # xgb.plot_importance(model)
    print('一共用时{}秒'.format(time.clock() - t0))



    # predictors = ['larger1_ratio', 'short3_ratio', 'short3', 'knn_min_values', 'rua12', 'rua11', 'rua10', 'mostpowerin',
    #               'mostcountin', 'wifimostcount', 'knn_std_groupRate', 'knn_std', 'knn_mean', 'rua9', 'rua8ratio',
    #               'rua8', 'shophourcount', 'top_count_rt', 'topcount', 'avgcossimilarity', 'maxcossimilarity',
    #               'powersmallest', 'wifiprosum_groupRate',
    #               'knn_max_values_groupRate', 'wifi_count_rt_groupRate', 'scorewifi_groupRate', 'rua5', 'rua6', 'rua7',
    #               'sryRate_groupRate',
    #               'bsRate_groupRate', 'srxRate_groupRate', 'dsRate_groupRate', 'ds_groupRate', 'sr_groupRate',
    #               'sryRate', 'bsRate', 'srxRate',
    #               'dsRate', 'ds', 'sr', 'wifitopcount', 'rua4', 'rua3', 'wifiinrat', 'wifinoshow', 'wifipromulti',
    #               'rua2', 'short2', 'rua1', 'wifiturerat',
    #               'larger2', 'larger1', 'short1', 'scorewifi', 'wifiprosum', 'h_s', 'time_dif', 'h_s_v', 'h_s_m',
    #               'knn_max_values', 'wifi_count_rt',
    #               'wifishopcount', 'h3', 'knn_values', 'real_lon_dif', 'real_lat_dif', 'w3', 'lon_real', 'lat_real',
    #               'real_ang_df', 'real_dis_df', 'connect_sum',
    #               'ang_df', 'dis_df', 'mall_id', 'user_id', 'dayofweek', 'hourofday', 'connect1', 'connect2',
    #               'power_dif2', 'ave_power2', 'power_dif1',
    #               'ave_power1', 'wifi2_power', 'wifi1_power', 'user_kind_count', 'category', 'user_shop_count',
    #               'latitude_s', 'longitude_s',
    #               'shop_hot', 'user_count']