
# -*- coding:utf-8 -*-

"""
@author: SJ
@software: PyCharm
@file: average_all.py 对train_2016_v2.csv文件中所有logerror进行平均，填充到sample_submission.csv处
@time: 2017-9-16 15:24
"""

import pandas as pd
import numpy as np

if __name__ == '__main__' :
    train = pd.read_csv("../../data/train_2016_v2.csv", parse_dates=['transactiondate'])
    avg = np.average(train['logerror'])

    predictlabel = pd.read_csv("../../data/sample_submission.csv", index_col='ParcelId')
    cols = predictlabel.columns
    print predictlabel.shape
    for c in cols:
        if str(c).startswith('2017'):
            continue
        # if c=='ParcelId':
        #     continue
        predictlabel[c] = avg
    print predictlabel.head()
    predictlabel.to_csv("./all_avg_base")