import pandas as pd

df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
print df.drop("a", axis=1)