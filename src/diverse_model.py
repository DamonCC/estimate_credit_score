from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb


f = open('../result/result.txt','w')



def lgb1_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_lgb = np.zeros(len(x_test))
    oof = np.zeros(len(x_train))
    seeds = [2019, 2019 * 2 + 1024, 4096, 2048, 1024]
    for model_seed in range(num_model_seed):
        param = {'num_leaves': 31, #一科树上的叶子数
                 'min_data_in_leaf': 20,#一个叶子上数据的最小数量。可以用来处理过拟合。
                 'objective': 'regression_l1',
                 'max_depth': 5,
                 'learning_rate': 0.0081,
                 "min_child_samples": 30,#用于一个叶子节点的最小样本数（数据集）. 每个叶子的较大样本量将减少过拟合（但可能导致欠拟合）
                 "boosting": "gbdt",#要用的算法
                 "feature_fraction": 0.7,# 每次迭代中，随机选择70%的特征进行迭代
                 "bagging_freq": 1,
                 "bagging_fraction": 0.8,#每次迭代时用的数据比例，类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
                 "bagging_seed": 11,
                 "metric": 'mae',#评价指标
                 "lambda_l1": 0.60,#L1正则化的lambda
                 "verbosity": -1}
        folds = KFold(n_splits=6, shuffle=True, random_state=seeds[0])
        oof_lgb = np.zeros(len(x_train))

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print(len(trn_idx))
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(x_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(x_train[val_idx], y_train[val_idx])
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                            early_stopping_rounds=500)
            oof_lgb[val_idx] = clf.predict(x_train[val_idx], num_iteration=clf.best_iteration)
            predictions_lgb += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits / num_model_seed
        oof += oof_lgb / num_model_seed
        MAE = mean_absolute_error(oof_lgb, y_train)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))
    MAE = mean_absolute_error(oof, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))
    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_lgb
    sub_df['score2'] = oof_lgb
    sub_df.to_csv('../result/lgb1_model_{}.csv'.format(name), index=0, header=1, sep=',')
    f.write("lgb1_model_{}---score: {:<8.8f}".format(name,score))
    f.write('\n')

def lgb2_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_lgb = np.zeros(len(x_test))
    oof = np.zeros(len(x_train))
    seeds = [2019, 2019 * 2 + 1024, 4096, 2048, 1024]
    for model_seed in range(num_model_seed):
        param = {'num_leaves': 31,
                 'min_data_in_leaf': 20,
                 'objective': 'regression_l2',
                 'max_depth': 5,
                 'learning_rate': 0.0081,
                 "min_child_samples": 30,
                 "boosting": "gbdt",
                 "feature_fraction": 0.7,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.8,
                 "bagging_seed": 11,
                 "metric": 'mae',
                 "lambda_l1": 0.60,
                 "verbosity": -1}
        folds = KFold(n_splits=6, shuffle=True, random_state=seeds[0])  #
        oof_lgb = np.zeros(len(x_train))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print(len(trn_idx))
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(x_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(x_train[val_idx], y_train[val_idx])
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                            early_stopping_rounds=500)
            oof_lgb[val_idx] = clf.predict(x_train[val_idx], num_iteration=clf.best_iteration)
            predictions_lgb += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits / num_model_seed
        oof += oof_lgb / num_model_seed
        MAE = mean_absolute_error(oof_lgb, y_train)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))
    MAE = mean_absolute_error(oof, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))

    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_lgb
    sub_df['score2'] = oof_lgb
    sub_df.to_csv('../result/lgb2_model_{}.csv'.format(name), index=0, header=1, sep=',')
    f.write("lgb2_model_{}---score: {:<8.8f}".format(name, score))
    f.write('\n')

def xgb1_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_xgb = np.zeros(len(x_test))
    oof_xgb = np.zeros(len(x_train))
    seeds = [2019, 4096, 2019 * 2 + 1024, 2048, 1024]
    for seed in range(num_model_seed):
        '''eta类似学习率，当这个参数值为1时，静默模式开启，不会输出任何信息。 一般这个参数就保持默认的0，因为这样能帮我们更好地理解模型
        subsample，这个参数控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合
        alpha权重的L1正则化项。
        colsample_bytree用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1
        lambda[默认1]权重的L2正则化项
        '''
        xgb_params = {'eta': 0.004, 'max_depth': 6, 'subsample': 0.5, 'colsample_bytree': 0.5, 'alpha': 0.2,
                      'objective': 'reg:gamma', 'eval_metric': 'mae', 'silent': True, 'nthread': -1
                      }
        folds = KFold(n_splits=5, shuffle=True, random_state=seeds[seed])
        oof = np.zeros(len(x_train))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print("fold n°{}".format(fold_ + 1))
            trn_data = xgb.DMatrix(x_train[trn_idx], y_train[trn_idx])
            val_data = xgb.DMatrix(x_train[val_idx], y_train[val_idx])

            watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
            clf = xgb.train(dtrain=trn_data, num_boost_round=10000, evals=watchlist, early_stopping_rounds=200,
                            verbose_eval=1000, params=xgb_params)
            oof[val_idx] = clf.predict(xgb.DMatrix(x_train[val_idx]), ntree_limit=clf.best_ntree_limit)
            predictions_xgb += clf.predict(xgb.DMatrix(x_test),
                                           ntree_limit=clf.best_ntree_limit) / folds.n_splits / num_model_seed
        oof_xgb += oof / num_model_seed
        MAE = mean_absolute_error(y_train, oof)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))

    MAE = mean_absolute_error(oof_xgb, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))

    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_xgb
    sub_df['score2'] = oof_xgb
    sub_df.to_csv('../result/xgb1_model_{}.csv'.format(name), index=0, header=1, sep=',')
    f.write("xgb1_model_{}---score: {:<8.8f}".format(name, score))
    f.write('\n')

def xgb2_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_xgb = np.zeros(len(x_test))
    oof_xgb = np.zeros(len(x_train))
    seeds = [2019, 4096, 2019 * 2 + 1024, 2048, 1024]
    for seed in range(num_model_seed):
        xgb_params = {'eta': 0.004, 'max_depth': 8, 'subsample': 0.5, 'colsample_bytree': 0.6, 'alpha': 0.1,
                      'objective': 'reg:gamma', 'eval_metric': 'mae', 'silent': True, 'nthread': -1
                      }
        folds = KFold(n_splits=5, shuffle=True, random_state=seeds[seed])
        oof = np.zeros(len(x_train))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print("fold n°{}".format(fold_ + 1))
            trn_data = xgb.DMatrix(x_train[trn_idx], y_train[trn_idx])
            val_data = xgb.DMatrix(x_train[val_idx], y_train[val_idx])

            watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
            clf = xgb.train(dtrain=trn_data, num_boost_round=10000, evals=watchlist, early_stopping_rounds=200,
                            verbose_eval=1000, params=xgb_params)
            oof[val_idx] = clf.predict(xgb.DMatrix(x_train[val_idx]), ntree_limit=clf.best_ntree_limit)
            predictions_xgb += clf.predict(xgb.DMatrix(x_test),
                                           ntree_limit=clf.best_ntree_limit) / folds.n_splits / num_model_seed
        oof_xgb += oof / num_model_seed
        MAE = mean_absolute_error(y_train, oof)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))

    MAE = mean_absolute_error(oof_xgb, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))

    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_xgb
    sub_df['score2'] = oof_xgb
    sub_df.to_csv('../result/xgb2_model_{}.csv'.format(name), index=0, header=1, sep=',')
    f.write("xgb2_model_{}---score: {:<8.8f}".format(name, score))
    f.write('\n')


def lgb3_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_lgb = np.zeros(len(x_test))
    oof = np.zeros(len(x_train))
    seeds = [2019, 2019 * 2 + 1024, 4096, 2048, 1024]
    for model_seed in range(num_model_seed):
        param = {'num_leaves': 31,
                 'min_data_in_leaf': 20,
                 'objective': 'regression_l1',
                 'max_depth': 5,
                 'learning_rate': 0.01,
                 "min_child_samples": 30,
                 "boosting": "gbdt",
                 "feature_fraction": 0.45,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.8,
                 "bagging_seed": 11,
                 "metric": 'mae',
                 "lambda_l1": 0.60,
                 "verbosity": -1}
        folds = KFold(n_splits=6, shuffle=True, random_state=seeds[0])  #
        oof_lgb = np.zeros(len(x_train))

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print(len(trn_idx))
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(x_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(x_train[val_idx], y_train[val_idx])
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                            early_stopping_rounds=500)
            oof_lgb[val_idx] = clf.predict(x_train[val_idx], num_iteration=clf.best_iteration)
            predictions_lgb += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits / num_model_seed

        oof += oof_lgb / num_model_seed
        MAE = mean_absolute_error(oof_lgb, y_train)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))
    MAE = mean_absolute_error(oof, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))

    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_lgb
    sub_df['score2'] = oof_lgb
    sub_df.to_csv('../result/lgb3_model_{}.csv'.format(name), index=False)
    f.write("lgb3_model_{}---score: {:<8.8f}".format(name, score))
    f.write('\n')

def lgb4_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_lgb = np.zeros(len(x_test))
    oof = np.zeros(len(x_train))
    seeds = [2018, 2019 * 2 + 1024, 4096, 2048, 1024]
    for model_seed in range(num_model_seed):
        param = {'num_leaves': 31,
                 'min_data_in_leaf': 20,
                 'objective': 'regression_l2',
                 'max_depth': 5,
                 'learning_rate': 0.01,
                 "min_child_samples": 30,
                 "boosting": "gbdt",
                 "feature_fraction": 0.45,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.8,
                 "bagging_seed": 11,
                 "metric": 'mae',
                 "lambda_l1": 0.60,
                 "verbosity": -1}
        folds = KFold(n_splits=6, shuffle=True, random_state=seeds[0])
        oof_lgb = np.zeros(len(x_train))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print(len(trn_idx))
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(x_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(x_train[val_idx], y_train[val_idx])
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                            early_stopping_rounds=500)
            oof_lgb[val_idx] = clf.predict(x_train[val_idx], num_iteration=clf.best_iteration)
            predictions_lgb += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits / num_model_seed

        oof += oof_lgb / num_model_seed
        MAE = mean_absolute_error(oof_lgb, y_train)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))
    MAE = mean_absolute_error(oof, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))

    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_lgb
    sub_df['score2'] = oof_lgb
    sub_df.to_csv('../result/lgb4_model_{}.csv'.format(name), index=False)
    f.write("lgb4_model_{}---score: {:<8.8f}".format(name, score))
    f.write('\n')





