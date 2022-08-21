# %%

import pandas as pd

data = pd.read_csv("../Products.csv", lineterminator="\n", index_col=0)

data.price = data.price.str.replace("£", "").str.replace(",", "").astype(float)
data.create_time = pd.to_datetime(data.create_time)
print(data.dtypes)

data.to_csv("../Products_cleaned.csv")
# %%
