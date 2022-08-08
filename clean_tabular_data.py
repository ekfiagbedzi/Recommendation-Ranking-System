import pandas as pd

data = pd.read_csv("Products.csv", lineterminator="\n")
data

data.describe(include="object")
