# -*- coding:utf-8 -*-

"""
@author: SJ
@software: PyCharm
@file: tongji.py
@time: 2017-9-14 11:40
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# in_frame = pd.read_csv("../data/properties_2016.csv", dtype=np.str)
# lable_frame = pd.read_csv("../data/train_2016_v2.csv", dtype=np.str)
# print "train_2016_v2:" + str(lable_frame.shape)
# print "properties_2016:" + str(in_frame.shape)
# print lable_frame["parcelid"].value_counts()
# print (in_frame.set_index("parcelid").join(lable_frame.set_index("parcelid"), how='right')).shape
# print (in_frame.set_index("parcelid").join(lable_frame.set_index("parcelid"), how='left')).shape
# print (in_frame.set_index("parcelid").join(lable_frame.set_index("parcelid"), how='inner')).shape
# try:
#     print in_frame["censustractandblock"].dropna()
# except KeyError:
#     print "没有该key值"
# print "-----------------------------------------------"
# print in_frame.set_index("parcelid").join(lable_frame.set_index("parcelid"), how='inner').shape
# in_frame.set_index("parcelid").join(lable_frame.set_index("parcelid"), how='inner').to_csv("../data/join_train_2016")


# 统计join后数据的月份交易量分布

joined_frame = pd.read_csv("../data/join_train_2016", dtype=np.str, parse_dates=["transactiondate"])

joined_frame[joined_frame["transactiondate"].dt.month>=10].set_index("parcelid").to_csv("../data/test_set")
joined_frame[joined_frame["transactiondate"].dt.month<10].set_index("parcelid").to_csv("../data/training_set")

date_count = joined_frame["transactiondate"].dt.month.value_counts()
print np.sum(date_count[date_count.index>=10])

print date_count.index, date_count.values
plt.figure(figsize=(10, 6))
plt.bar(date_count.index, date_count.values)
plt.xlabel("month")
plt.ylabel("transaction num")

for x,y in zip(date_count.index, date_count.values):
    plt.text(x, y+0.1, '%d' % y, ha='center', va='bottom')

plt.show()


###############################################################

# 统计sample_submission

sample_frame = pd.read_csv("../data/sample_submission.csv")
print sample_frame["ParcelId"].value_counts()