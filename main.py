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
from torchvision import transforms
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
        return F.softmax(self.resnet50(X), dim=1)


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



            print(
                "Batch Round {}: Train Loss = {} Train Accuracy = {}" \
                   .format(batch_ind, train_loss.item(), train_accuracy))
                    

            batch_ind += 1            

    return {
        "Epoch": epoch,
        "TrainLoss": train_loss.item(),
        "TrainAccuracy": train_accuracy}


if __name__ == "__main__":
    torch.cuda.empty_cache()
    batch_size = 32
    epochs = 30
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_pickle("data/tables/image_product.pkl")
   

    train_data = ImageDataset.load_data(data)
    train_loader = DataLoader(train_data, batch_size, True)

    model = TL()
    start_time = time.time()
    train_metrics = train(model, epochs)
    end_time = time.time()


    ts = int(time.time())
    os.mkdir("model_evaluation/{}/".format(ts))
    os.mkdir("model_evaluation/{}/weights/".format(ts))
    os.mkdir("model_evaluation/{}/metrics/".format(ts))
    
    # save model parameters
    torch.save(
        model.state_dict(),
        "model_evaluation/{}/weights/{}.pt".format(ts, epochs))
    
    # save label decoder
    with open("image_decoder.json", "a+") as f:
        json.dump(dict(enumerate(train_data.le.classes_)), f)

    # save model metrics
    with open(
        "model_evaluation/{}/metrics/{}.json".format(ts, epochs), "a+") as f:
        json.dump(
            {"TrainingTime": (end_time-start_time)/60,
            "BatchSize": batch_size,
            "Epochs": epochs,
            "train_metrics": train_metrics}, f)