import os
import time
import json
import tqdm
from utils.helpers import ImageDataset, ResNet50

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(model, epochs=10):

    writer = SummaryWriter()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_ind = 0
    validation_loss = 0
    for epoch in range(epochs):
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        model.train()
        for _, (features, labels) in progress_bar:
            optimizer.zero_grad()
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
            batch_ind += 1
            if validation_loss == 0:       
                progress_bar.set_description("Epoch {}: Train Loss = {} Train Accuracy = {}" \
                   .format(epoch+1, round(train_loss.item(), 2), round(train_accuracy, 2)))
            else:
                progress_bar.set_description("Epoch {}: Train Loss = {} Train Accuracy = {} Validation Loss = {} Validation Accuracy = {}" \
                   .format(epoch+1, round(train_loss.item(), 2), round(train_accuracy, 2), round(validation_loss.item(), 2), round(validation_accuracy, 2)))

   
        with torch.no_grad():
            model.eval()                
            features, labels = next(iter(validation_loader))
            features, labels = features.to(device), labels.to(device)
            predictions = model(features)
            validation_loss = F.cross_entropy(predictions, labels)
            predictions = torch.argmax(predictions, dim=1)
            validation_accuracy = metrics.accuracy_score(labels.cpu(), predictions.cpu())
                
                
            writer.add_scalar(
                "Validation Loss", validation_loss.item(), batch_ind)
            writer.add_scalar(
                "Validation Accuracy", validation_accuracy, batch_ind)       

            batch_ind += 1            
            
        
            
    return {
        "Epoch": epoch,
        "TrainLoss": train_loss.item(),
        "TrainAccuracy": train_accuracy,
        "ValidationLoss": validation_loss.item(),
        "ValidationAccuracy": validation_accuracy,
        }


def test(model):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        #features, labels = next(iter(test_loader))
        features, labels = features.to(device), labels.to(device)
        predictions = model(features)
        test_loss = F.cross_entropy(predictions, labels)
        predictions = torch.argmax(predictions, dim=1)
        test_accuracy = metrics.accuracy_score(labels.cpu(), predictions.cpu())
        print("Test Loss = {} Test Accuracy = {}".format(test_loss.item(), test_accuracy))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    batch_size = 128
    epochs = 60
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    data = ImageDataset("data/tables/image_product.pkl")
    train_data, test_data = train_test_split(data, test_size=0.10, shuffle=True)
    #validation_data, test_data = train_test_split(test_data, test_size=0.4, shuffle=True)

    train_loader = DataLoader(train_data, batch_size, True)
    #test_loader = DataLoader(test_data, len(test_data))
    validation_loader = DataLoader(test_data, len(test_data))


    model = ResNet50()
    for para in model.parameters():
        params = model.state_dict()
        print(params.keys())
        zzzz
    start_time = time.time()
    train_metrics = train(model, epochs)
    end_time = time.time()

    #test(model)


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
        json.dump(data.decoder, f)

    # save model metrics
    with open(
        "model_evaluation/{}/metrics/{}.json".format(ts, epochs), "a+") as f:
        json.dump(
            {"TrainingTime": (end_time-start_time)/60,
            "BatchSize": batch_size,
            "Epochs": epochs,
            "train_metrics": train_metrics}, f)