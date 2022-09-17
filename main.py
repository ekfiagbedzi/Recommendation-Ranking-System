import os
import time
from utils.helpers import ImageDataset

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class TL(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_resnet50',
            pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13)

    def forward(self, X):
        return F.softmax(self.resnet50(X))


def train(model, epochs=10):

    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    batch_ind = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            predictions = model(features)
            train_loss = F.cross_entropy(predictions, labels)
            train_loss.backward()
            print("Train Loss = {}".format(train_loss.item()))
            optimizer.step()
            writer.add_scalar("Train Loss", train_loss.item(), batch_ind)

        with torch.no_grad():
            model.eval()
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)
                predictions = model(features)
                validation_loss = F.cross_entropy(predictions, labels)
                print("Validation Loss = {}".format(validation_loss.item()))

                writer.add_scalar(
                    "Validation Loss", validation_loss.item(), batch_ind)

            batch_ind += 1            
        print("Epoch {}: Train Loss = {}, Validation Loss = {}".format(
                epoch+1, train_loss.item(), validation_loss.item()))

def test(model):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            predictions = model(features)
            test_loss = F.cross_entropy(predictions, labels)
            print("Test Loss = {}".format(test_loss.item()))


if __name__ == "__main__":
    epoch = 10
    data = pd.read_pickle("image_product.pkl")
    train_data, test_data = train_test_split(
        data, test_size=0.3, shuffle=True)
    validation_data, test_data = train_test_split(
        test_data, test_size=0.4, shuffle=True)

    train_data = ImageDataset.load_data(train_data)
    test_data = ImageDataset.load_data(test_data)
    validation_data = ImageDataset.load_data(validation_data)


    train_loader = DataLoader(train_data, 30, True)
    test_loader = DataLoader(test_data)
    validation_loader = DataLoader(validation_data)
    model = TL()
    train(model, epoch)

    ts = int(time.time())
    os.mkdir("model_evaluation/{}/".format(ts))
    os.mkdir("model_evaluation/{}/weights/".format(ts))
    torch.save(
        model.state_dict(),
        "model_evaluation/{}/weights/{}.pt".format(ts, epoch))
