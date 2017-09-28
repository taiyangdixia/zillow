# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv("../data/properties_2016.csv", dtype=object)
for col in df.columns:
    print col, len(pd.unique(df[col]))
