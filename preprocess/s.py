# -*- coding: utf-8 -*-

import pandas as pd

dic = {"哈哈": [0, 1, 2, 3],
       "state": ["run", "run", "down", "on"]
       }
df = pd.DataFrame(dic, index=["小明", "小红", "xiaohua", 4], columns=["哈哈", "state"])

print df
print df["state"]

print df.columns
