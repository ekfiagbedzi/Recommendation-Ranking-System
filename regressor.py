#%%
import pandas as pd


data = pd.read_csv("Products.csv", lineterminator="\n", index_col=0, usecols=["product_name", product_description])
data
# %%
