# -*- coding:utf-8 -*-

"""
@author: SJ
@software: PyCharm
@file: xgboost_trial.py
@time: 2017-9-15 10:38
"""

import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from feature.feature_process import feature_process

import xgboost as xgb


if __name__ == "__main__":

    drop_feature = []
    code_feature = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]

    # fp = feature_process()
    # 读取训练集数据
    train = pd.read_csv("../data/join_train_2016", parse_dates=["transactiondate"], low_memory=False)#, dtype={"hashottuborspa": np., propertycountylandusecode, propertyzoningdesc, fireplaceflag, taxdelinquencyflag"})
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
        n_estimators=200,
        objective='reg:linear',
        max_depth=5,
        eta=0.05,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric='mae',
        missing=-999)

    # col = train.columns
    # for c in col
    test = pd.read_csv("../data/properties_2016.csv", low_memory=False)
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
    xgtrain = xgb.DMatrix(train.values, target.values)

    cvresult = xgb.cv(regressor.get_xgb_params(), xgtrain, num_boost_round=regressor.get_params()["n_estimators"], nfold=5, folds=5, shuffle=True)

    print "best cv:"
    print cvresult[cvresult["test-mae-mean"] == np.min(cvresult["test-mae-mean"])]

    best_num = cvresult[cvresult["test-mae-mean"]==np.min(cvresult["test-mae-mean"])].index

    regressor.get_params()["n_estimators"] = best_num
    print regressor.get_params()

    print best_num

    regressor.fit(train, target, eval_metric='mae')

    print "fscore: ",
    feature_importance = pd.Series(regressor.get_booster().get_fscore()).sort_values(ascending=False)
    print feature_importance

    feature_importance.plot(kind="bar", title="Feature importances")
    plt.ylabel("feature importance score")


    testParcelid = test["parcelid"]

    # train = train.drop(labels=["censustractandblock", "logerror", "transactiondate"], axis=1)
    test = test.drop("parcelid", axis=1)

    print type(test.values)
    testDMatrix = test
    predict_result_tmp = regressor.predict(testDMatrix)
    predict_result_tmp = predict_result_tmp.astype(np.float64)

    # predict_result = np.absolute(predict_result)
    print predict_result_tmp.shape

    predict_result = np.around(predict_result_tmp, decimals=4)
    print predict_result[:2]

    temp2 = np.vstack((predict_result, predict_result))
    print temp2.shape
    temp3 = np.vstack((temp2, predict_result))
    print temp3.shape
    predict_result = np.transpose(np.vstack((temp3, temp3)))

    resultDF = pd.DataFrame(np.around(predict_result, decimals=4),
                            index=testParcelid,
                            columns=["201610", "201611", "201612", "201710", "201711", "201712"])
    resultDF.index.name = "ParcelId"
    resultDF.to_csv("../data/out/{}".format(datetime.now().strftime("%Y%m%d_%H_%M")),  float_format='%.4f')
    print "done"
