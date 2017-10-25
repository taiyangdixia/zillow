# -*- coding:utf-8 -*-

"""
@author: SJ
@software: PyCharm
@file: sns_heatmap.py
@time: 2017-10-11 16:05
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../../data/join_train_2016", parse_dates=["transactiondate"])
df["month"] = df["transactiondate"].dt.month

plt.figure(figsize=(20, 8))
sns.set()
feature = ['taxvaluedollarcnt',
'structuretaxvaluedollarcnt',
'landtaxvaluedollarcnt',
'taxamount',
'assessmentyear'
]
cor = df[feature].corr()
mask = np.zeros_like(cor)
mask[np.triu_indices_from(mask)] = True

# df.pivot("month", "yearbuilt", "logerror")
sns.heatmap(cor, mask=mask, vmin=0, vmax=1, annot=True)
plt.show()

