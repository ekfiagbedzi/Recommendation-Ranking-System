from utils.helpers import ImageDataset

import pandas as pd

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    batch_ind = 0
    for epoch in range(epochs):
        for batch in loader:
            features, labels = batch
            predictions = model(features)
            loss = F.cross_entropy(predictions, labels)
            loss.backward()
            print("Loss = {}".format(loss.item()))
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("Loss", loss.item(), batch_ind)
            batch_ind += 1            
        print("Epoch {}: Loss is {}".format(epoch+1, loss.item()))


if __name__ == "__main__":
    data = pd.read_pickle("image_product.pkl")
    image_data = ImageDataset.load_data(data)
    loader = DataLoader(image_data, 5, True)
    model = TL()
    train(model)

    torch.save(model.state_dict(), "model.pt")
