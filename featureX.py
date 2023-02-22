import os
import json
from image_processor import process_image
from utils.helpers import ResNet50

import numpy as np
import pandas as pd

import torch


if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # load data
    data = pd.read_pickle("data/tables/image_product.pkl")
    image_id = data.id.to_list()
    

    # define model
    model = ResNet50()
    model.load_state_dict(
        torch.load(
            "/home/biopythoncodepc/Documents/git_repositories/Recommendation-Ranking-System/model_evaluation/1677054300/weights/10.pt"))
    model.resnet50.fc = torch.nn.Linear(2048, 1000)
    embeddings = []
    
    # extract high level features
    with torch.no_grad():
        model.to(device)
        model.eval()
        for ID in image_id:
            feature = model(process_image(ID).to(device))
            embeddings.append((np.array(feature.cpu())).squeeze().tolist())

    with open("image_embeddings.json", "w") as f:
        diction = dict(zip(image_id, embeddings))
        json.dump(diction, f)