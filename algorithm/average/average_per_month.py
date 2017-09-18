
# -*- coding:utf-8 -*-

"""
@author: SJ
@software: PyCharm
@file: average_all.py

对在train_2016_v2.csv文件中的logerror根据月份进行平均计算，
各月份的平均值对应到sample_submission中的相应月份。

@time: 2017-9-16 15:24
"""

import pandas as pd
import numpy as np

if __name__ == '__main__' :
    train = pd.read_csv("../../data/train_2016_v2.csv", parse_dates=['transactiondate'])
    avg = np.average(train['logerror'])

    train["month"] = train["transactiondate"].dt.month
    train_avg_per_month = pd.DataFrame(train.groupby(["month"])['logerror'].agg(["mean"]))
    # temp = train.groupby(["month"])['logerror'].agg("count")
       #.avg('logerror')

    print train_avg_per_month
    predictlabel = pd.read_csv("../../data/sample_submission.csv", index_col='ParcelId')
    cols = predictlabel.columns
    # print predictlabel.shape
    print train_avg_per_month.index
    print train_avg_per_month.columns

    print type(train_avg_per_month)
    for c in cols:
        # if c=='ParcelId':
        #     continue
        # print train_avg_per_month['mean'].ix[int(c[4:6])]
        if str(c).startswith('2017'):
            continue
        predictlabel[c] = train_avg_per_month.ix[int(c[4:6]), u'mean']
    # print predictlabel.head()
    predictlabel.to_csv("./month_avg_base")