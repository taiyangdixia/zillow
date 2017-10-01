# -*- coding:utf-8 -*-

"""
@author: SJ
@software: PyCharm
@file: xgboost_trial.py
@time: 2017-9-15 10:38
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

import xgboost as xgb


if __name__ == "__main__":

    drop_feature = []
    code_feature = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]

    # fp = feature_process()
    # 读取训练集数据
    train = pd.read_csv("../../data/join_train_2016", parse_dates=["transactiondate"], low_memory=False)#, dtype={"hashottuborspa": np., propertycountylandusecode, propertyzoningdesc, fireplaceflag, taxdelinquencyflag"})
    train = train.drop(drop_feature, axis=1)
    print train.shape

    # 去除一些预测不准的点，尽量拟合logerror比较小的点
    train = train[train["logerror"] > -0.4]
    train = train[train["logerror"] < 0.419]
    # train = fp.process(train)

    # 训练数据空值填充-999
    train = train.fillna(-999)

    print "train shape:", train.shape

    # train = train.drop(labels=["censustractandblock", "logerror", "transactiondate"], axis=1)

    train = train.drop("transactiondate", axis=1)

    regressor = xgb.XGBRegressor(
        n_estimators=1000,
        objective='reg:linear',
        max_depth=5,
        eta=0.05,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric='mae',
        missing=-999)

    # col = train.columns
    # for c in col
    test = pd.read_csv("../../data/properties_2016.csv", low_memory=False)
    test = test.drop(drop_feature, axis=1)
    # test = fp.process(test)

    test = test.fillna(-999)

    print "test shape:", test.shape
    for col in code_feature:
        if test[col].dtype == "object":
            encoder = LabelEncoder()
            encoder.fit(test[col])
            train[col] = encoder.transform(train[col])
            test[col] = encoder.transform(test[col])

    target = train['logerror']
    print train.shape
    train = train.drop(["parcelid", "logerror"], axis=1)
    # xgtrain = xgb.DMatrix(train.values, target.values)

    # cvresult = xgb.cv(regressor.get_xgb_params(), xgtrain, num_boost_round=regressor.get_params()["n_estimators"], nfold=5, folds=5, shuffle=True)
    param_test1 = {
        'max_depth': range(3, 7, 1),
        'min_child_weight': range(1, 6, 2)
    }

    gridSearch = GridSearchCV(regressor, param_grid=param_test1,
                              scoring='neg_mean_absolute_error',
                              n_jobs=4, iid=False, cv=5)

    gridSearch.fit(train, target)

    print gridSearch.grid_scores_
    print gridSearch.best_params_
    print gridSearch.best_score_
    print "done"
