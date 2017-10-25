import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/join_train_2016", parse_dates=["transactiondate"])
df["month"] = df["transactiondate"].dt.month

sns.set(style="ticks")
sns.boxplot(x="month", y="yearbuilt", hue="fireplaceflag", data=df)
plt.show()
print df.columns
df["parcelid"] = pd.factorize(df["parcelid"], sort=True)[0]+1

print df
print type(df)
print df["fireplaceflag"].value_counts()