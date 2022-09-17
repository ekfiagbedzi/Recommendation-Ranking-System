import os
import time
import json
from utils.helpers import ImageDataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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


def train(model, epochs=10):

    writer = SummaryWriter()
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
            predictions = torch.argmax(predictions, dim=1)
            train_accuracy = metrics.accuracy_score(
                labels.cpu(), predictions.cpu())
            optimizer.step()
            writer.add_scalar("Train Loss", train_loss.item(), batch_ind)
            writer.add_scalar("Train Accuracy", train_accuracy, batch_ind)


            with torch.no_grad():
                model.eval()                
                features, labels = next(iter(validation_loader))
                features, labels = features.to(device), labels.to(device)
                predictions = model(features)
                validation_loss = F.cross_entropy(predictions, labels)
                predictions = torch.argmax(predictions, dim=1)
                validation_accuracy = metrics.accuracy_score(
                    labels.cpu(), predictions.cpu())

                print(
                    "Batch Round {}: Train Loss = {} Train Acc = {} Val Loss = \
                        {} Val Acc = {}".format(
                            batch_ind,
                            train_loss.item(),
                            train_accuracy,
                            validation_loss.item(),
                            validation_accuracy))
                
                writer.add_scalar(
                    "Validation Loss", validation_loss.item(), batch_ind)
                writer.add_scalar(
                    "Validation Accuracy", validation_accuracy, batch_ind)
                    

            batch_ind += 1            
        print(
            "Epoch {}: Train Loss = {} Train Acc = {} Val Loss = {} Val Acc = \
                {}".format(
                epoch+1,
                train_loss.item(),
                train_accuracy,
                validation_loss.item(),
                validation_accuracy))
    return {
        "Epoch": epoch,
        "TrainLoss": train_loss.item(),
        "TrainAccuracy": train_accuracy,
        "ValidationLoss": validation_loss.item(),
        "ValidationAccuracy": validation_accuracy}

def test(model):
    with torch.no_grad():
        model.to(device)
        model.eval()
        features, labels = next(iter(test_loader))
        features, labels = features.to(device), labels.to(device)
        predictions = model(features)
        test_loss = F.cross_entropy(predictions, labels)
        predictions = torch.argmax(predictions, dim=1)
        test_accuracy = metrics.accuracy_score(labels.cpu(), predictions.cpu())
        print(
            "Test Loss = {} Test Accuracy = {}".format(
                test_loss.item(), test_accuracy))
    return {
        "TestLoss": test_loss.item(),
        "TestAccuracy": test_accuracy}


if __name__ == "__main__":
    epochs = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_pickle("image_product.pkl")
    train_data, test_data = train_test_split(
        data, test_size=0.3, shuffle=True)
    validation_data, test_data = train_test_split(
        test_data, test_size=0.4, shuffle=True)

    train_data = ImageDataset.load_data(train_data)
    test_data = ImageDataset.load_data(test_data)
    validation_data = ImageDataset.load_data(validation_data)


    train_loader = DataLoader(train_data, 32, True)
    test_loader = DataLoader(test_data, len(test_data))
    validation_loader = DataLoader(validation_data, len(validation_data))

    model = NN()
    train_metrics = train(model, epochs)
    test_metrics = test(model)

    ts = int(time.time())
    os.mkdir("model_evaluation/{}/".format(ts))
    os.mkdir("model_evaluation/{}/weights/".format(ts))
    os.mkdir("model_evaluation/{}/metrics/".format(ts))

    torch.save(
        model.state_dict(),
        "model_evaluation/{}/weights/{}.pt".format(ts, epochs))

    with open(
        "model_evaluation/{}/metrics/{}.json".format(ts, epochs), "a+") as f:
        json.dump(
            {"train_metrics": train_metrics, "test_metrics": test_metrics}, f)