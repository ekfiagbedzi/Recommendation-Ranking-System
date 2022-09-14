import typing
from utils.helpers import get_element, ImageData

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import DataLoader


class NN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 13),
            torch.nn.Softmax()
        )


if __name__ == "main":
   
    le = LabelEncoder()
    data = pd.read_pickle("image_product.pkl")
    X = data.image_array
    y = le.fit_transform(
        data.category.str.split("/").apply(get_element, position=0))
    image_data = ImageData(X, y)
    loader = DataLoader(image_data, 5, True)
