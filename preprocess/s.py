# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv("../data/properties_2016.csv")
print df.shape

df_res = pd.read_csv("../data/sample_submission.csv")
print df_res.shape

df2 = pd.read_csv("../data/join_train_2016")
print df2.columns
