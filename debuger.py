import os
from pickletools import optimize
import time
import json
import tqdm
from utils.helpers import CombinedDataset, ImageDataset

import pandas as pd
from sklearn import metrics

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class ImageClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_resnet50',
            pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13)
        
    def forward(self, X):
        return self.resnet50(X)


class TextClassifier(torch.nn.Module):
    def __init__(self, input_size: int = 768):
        super(TextClassifier, self).__init__()
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
        return self.layers(X)




class EnsembleModelPreTrained(torch.nn.Module):
    def __init__(self, text_model, image_model) -> None:
        super(EnsembleModelPreTrained, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.output_layer = torch.nn.Linear(26, 13)

        
    def forward(self, image_features, text_features):
        text_predictions = self.text_model(text_features)
        image_predictions = self.image_model(image_features)
        features = torch.cat((text_predictions, image_predictions), dim=1)
        return F.softmax(self.output_layer(features), dim=1)


def combined_train(model, epochs=10):

    writer = SummaryWriter()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    batch_ind = 0
    for epoch in range(epochs):
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for _, ((image_features, text_features), labels) in progress_bar:
            optimizer.zero_grad()
            image_features, text_features, labels = image_features.to(device), text_features.to(device), labels.to(device)
            predictions = model(image_features, text_features)
            train_loss = F.cross_entropy(predictions, labels)
            train_loss.backward()
            predictions = torch.argmax(predictions, dim=1)
            train_accuracy = metrics.accuracy_score(
                labels.cpu(), predictions.cpu())
            optimizer.step()
            writer.add_scalar("Train Loss", train_loss.item(), batch_ind)
            writer.add_scalar("Train Accuracy", train_accuracy, batch_ind)
                    

            batch_ind += 1            
            progress_bar.set_description("Epoch {}: Train Loss = {} Train Accuracy = {}" \
                   .format(epoch+1, round(train_loss.item(), 2), round(train_accuracy, 2)))

    return {
        "Epoch": epoch,
        "TrainLoss": train_loss.item(),
        "TrainAccuracy": train_accuracy}


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = CombinedDataset("data/tables/image_product.pkl")
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    text_model = TextClassifier()
    image_model = ImageClassifier()
    text_model.load_state_dict(torch.load("final_models/text_model.pt"))
    image_model.load_state_dict(torch.load("final_models/image_model.pt"))
    model = EnsembleModelPreTrained(text_model, image_model)
    combined_train(model)