# -*- coding:utf-8 -*-

"""
@author: SJ
@software: PyCharm
@file: xgb_normal.py
@time: 2017-9-27 21:23

"""

import numpy as np
import pandas as pd
import xgboost as xgb

train = pd.read_csv("../data/train_2016_v2.csv")
properties = pd.read_csv("../data/properties_2016.csv")
sample = pd.read_csv("../data/sample_submission.csv")

label = ["logerror"]
index = ["parcelid"]

for col in properties.columns:
    properties[col]=properties[col].fillna(-1)



