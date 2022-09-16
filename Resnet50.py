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
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
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
        for batch in loader:
            optimizer.zero_grad()
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            predictions = model(features)
            loss = F.cross_entropy(predictions, labels)
            loss.backward()
            #print("Loss = {}".format(loss.item()))
            optimizer.step()
            writer.add_scalar("Loss", loss.item(), batch_ind)

            with torch.no_grad():
                for features, labels in loader:
                    features, labels = features.to(device), labels.to(device)
                    predictions = model(features)
                    loss = F.cross_entropy(predictions, labels)
                    writer.add_scalar("Loss", loss.item(), batch_ind)

            batch_ind += 1            
        print("Epoch {}: Loss is {}".format(epoch+1, loss.item()))


if __name__ == "__main__":
    epoch = 10
    data = pd.read_pickle("image_product.pkl")
    train_data, test_data = train_test_split(data, test_size=0.3, shuffle=True)
    validation_data, test_data = train_test_split(test_data, test_size=0.4, shuffle=True)


    dddd
    #image_data = ImageDataset.load_data(data)


    dddd
    loader = DataLoader(image_data, 5, True)
    model = TL()
    train(model, epoch)

    ts = int(time.time())
    os.mkdir("model_evaluation/{}/".format(ts))
    os.mkdir("model_evaluation/{}/weights/".format(ts))
    torch.save(model.state_dict(), "model_evaluation/{}/weights/{}.pt".format(ts, epoch))
