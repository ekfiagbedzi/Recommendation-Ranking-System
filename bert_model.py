from utils.helpers import TextDataSet

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
class TextClassifier(nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 13))

    def forward(self, X):
        return F.softmax(self.layers(X), dim=1)

def train(model, epochs=20):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        model.train()
        for features, labels in dataloader:
            optimizer.zero_grad()
            features, labels = features.to(device), labels.to(device)
            predictions = model(features)
            loss = F.cross_entropy(predictions, labels)
            print(loss)
            loss.backward()
            optimizer.step()
            print(loss)

if __name__ == "__main__":
    dataset = TextDataSet()
    dataloader = DataLoader(dataset, batch_size=24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier()
    train(model, epochs=2)

