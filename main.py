import typing
from utils.helpers import get_element, ImageDataset

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
            torch.nn.Conv2d(3, 16, 7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 7),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(8192, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 13),
            torch.nn.Softmax(1)
        )


    def forward(self, X):
        return self.layers(X)


def train(model, features, targets):
    predictions = model(features)
    loss = F.cross_entropy(predictions, targets)
    loss.backward()
    print("Loss is", loss.item())

if __name__ == "__main__":
   
    data = pd.read_pickle("image_product.pkl")
    image_data = ImageDataset.load_data(data)
    loader = DataLoader(image_data, 5, True)
    features, labels = next(iter(loader))
    model = NN()
    train(model, features, labels)
