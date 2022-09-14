# %%
from utils.helpers import get_element, ImageData
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import torch

# %%
le = LabelEncoder()
# %%
data = pd.read_pickle("image_product.pkl")
data


# %%
X = data.image_array
y = le.fit_transform(
        data.category.str.split("/").apply(get_element, position=0))
# %%
image_data = ImageData(X, y)
# %%
