import os
import time
import json
import tqdm
from utils.helpers import CombinedDataset, EnsembleArchitecture, EnsemblePreTrained, TextClassifier, ResNet50

from sklearn import metrics

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(model, epochs=10):

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
    epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = CombinedDataset("data/tables/image_product.pkl")
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    text_model = TextClassifier()
    image_model = ResNet50()
    text_model.load_state_dict(torch.load("final_models/text_model.pt"))
    image_model.load_state_dict(torch.load("final_models/image_model.pt"))
    model = EnsemblePreTrained(text_model, image_model)
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
    with open("combined_decoder.json", "a+") as f:
        json.dump(train_data.decoder, f)

    # save model metrics
    with open(
        "model_evaluation/{}/metrics/{}.json".format(ts, epochs), "a+") as f:
        json.dump(
            {"TrainingTime": (end_time-start_time)/60,
            "BatchSize": batch_size,
            "Epochs": epochs,
            "train_metrics": train_metrics}, f)