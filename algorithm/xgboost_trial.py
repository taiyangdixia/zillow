# -*- coding:utf-8 -*-

"""
@author: SJ
@software: PyCharm
@file: xgboost_trial.py
@time: 2017-9-15 10:38
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

if __name__ == "__main__":

    # 读取训练集数据
    train = pd.read_csv("../data/join_train_2016", parse_dates=["transactiondate"], low_memory=False)#, dtype={"hashottuborspa": np., propertycountylandusecode, propertyzoningdesc, fireplaceflag, taxdelinquencyflag"})

    # 训练数据空值填充-999
    train = train.fillna(-999)

    print "train shape:", train.shape

    # train = train.drop(labels=["censustractandblock", "logerror", "transactiondate"], axis=1)

    train = train.drop("transactiondate", axis=1)

    regressor = xgb.XGBRegressor(n_jobs=4,
        n_estimators=200,
        objective='reg:linear',
        max_depth=5,
        learning_rate=0.2,
        min_child_weight=2,
        eval_metric='mae',
        missing=-999)

    # col = train.columns
    # for c in col:
    encoder = LabelEncoder()

    for col in ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]:
        train[col] = encoder.fit_transform(train[col])


    params = {
        'n_jobs': 4,
        'n_estimators': 200, # 多少个树
        'objective': 'reg:linear',
        'max_depth': 5,
        'eta': 0.2,
        'min_child_weight': 2,
        'eval_metric': 'mae',
        'missing': -999
    }

    target = train['logerror']
    print train.columns
    train = train.drop(["parcelid", "logerror"], axis=1)
    xgtrain = xgb.DMatrix(train.values, target.values)

    # cvresult = xgb.cv(regressor.get_xgb_params(), xgtrain, num_boost_round=regressor.get_params()["n_estimators"], nfold=5, folds=5, shuffle=True)

    # print "cv result => ", cvresult
    regressor.fit(train, target, eval_metric='mae')

    test = pd.read_csv("../data/properties_2016.csv", low_memory=False)
    test = test.fillna(-999)

    for col in ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]:
        test[col] = encoder.fit_transform(test[col])

    print "test shape:", test.shape

    testParcelid = test["parcelid"]
    # train = train.drop(labels=["censustractandblock", "logerror", "transactiondate"], axis=1)
    test = test.drop("parcelid", axis=1)

    print type(test.values)
    testDMatrix = test
    predict_result = regressor.predict(testDMatrix)

    stacked_result = np.hstack((testParcelid, predict_result))
    print stacked_result.shape
    type(stacked_result)