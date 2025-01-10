import os
import time
import json
from tqdm import tqdm
from utils.helpers import TextDataSet, TextClassifier

from sklearn import metrics

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils. tensorboard import SummaryWriter


def train(model, epochs=10):

    writer = SummaryWriter()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_ind = 0
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, (features, labels) in progress_bar:
            optimizer.zero_grad()
            features, labels = features.to(device), labels.to(device)
            predictions = model(features, False)
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
    batch_size = 32
    epochs = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

    data = TextDataSet()
    dataloader = DataLoader(data, batch_size, True)

    model = TextClassifier()
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
    with open("text_decoder.json", "a+") as f:
        json.dump(data.decoder, f)

    # save model metrics
    with open(
        "model_evaluation/{}/metrics/{}.json".format(ts, epochs), "a+") as f:
        json.dump(
            {"TrainingTime": (end_time-start_time)/60,
            "BatchSize": batch_size,
            "Epochs": epochs,
            "train_metrics": train_metrics}, f)

