# -*- coding:utf-8 -*-

"""
@author: SJ
@software: PyCharm
@file: lightGBM.py
@time: 2017-10-11 17:40
"""

from datetime import datetime
import numpy as np
import pandas as pd

import lightgbm as lgb

import matplotlib.pyplot as plt

train = pd.read_csv("../data/train_2016_v2.csv")
prop = pd.read_csv("../data/properties_2016.csv")

for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how="left", on="parcelid")

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate',
                         'propertyzoningdesc', 'propertycountylandusecode'],
                        axis=1)
y_train = df_train["logerror"].values
print x_train.shape, y_train.shape, type(df_train["logerror"]), type(y_train)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

split = 90000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

print type(x_train), type(x_valid)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

params = {}

params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

watchlist = [d_valid]
clf =  lgb.train(params, d_train, 560, watchlist)

print "clf best score=>", clf.best_score

sample = pd.read_csv("../data/sample_submission.csv")
sample["parcelid"] = sample["ParcelId"]
df_test = sample.merge(prop, how="left", on="parcelid")

x_test = df_test[train_columns]

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

x_test = x_test.values.astype(np.float32, copy=False)

p_test = clf.predict(x_test)

predict_result = np.around(p_test, decimals=4)
print predict_result[:2]

temp2 = np.vstack((predict_result, predict_result))
print temp2.shape
temp3 = np.vstack((temp2, predict_result))
print temp3.shape
predict_result = np.transpose(np.vstack((temp3, temp3)))

resultDF = pd.DataFrame(predict_result, index=sample["ParcelId"],
             columns=["201610", "201611", "201612", "201710", "201711", "201712"])

resultDF.to_csv("../data/out/{}".format(datetime.now().strftime("%Y%m%d_%H_%M")), float_format='%.4f')
print "done"
