import typing
from utils.helpers import get_element, ImageData

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FF
from torch.utils.data import DataLoader



class NN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2352, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 13),
            torch.nn.Softmax()
        )


    def forward(self, X):
        return self.layers(X)


def train(model, features, targets):
    predictions = model(features)
    loss = F.cross_entropy(predictions, targets)
    loss.backward()
    print("Loss is", loss.item())

if __name__ == "__main__":
   
    le = LabelEncoder()
    data = pd.read_pickle("image_product.pkl")
    X = data.image_array
    y = le.fit_transform(
        data.category.str.split("/").apply(get_element, position=0))
    image_data = ImageData(X, y)
    loader = DataLoader(image_data, 5, True)
    features, labels = next(iter(loader))
    features = FF.to_tensor(features)
    labels = FF.to_tensor(labels)
    model = NN()
    train(model, F.tofeatures, labels)
    
