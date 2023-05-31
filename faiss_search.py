import os
import json
from image_processor import image_processor
from utils.helpers import ResNet50

import numpy as np
import pandas as pd
from PIL import Image
from faiss import read_index

import torch


if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # load data
    with open("image_embeddings.json", "r") as f:
        data = json.load(f)
    ids, embeddings = list(data.keys()), list(data.values())
    vectors = pd.DataFrame({"id": ids, "embeddings": embeddings})
    data = pd.read_pickle("data/tables/image_product.pkl")
    categories = data.loc[:, ["id", "category"]]
    reference = pd.merge(vectors, categories, on="id")
    

    # create search vector
    pil_image = Image.open("dog.jpg")
    index = read_index("rrs.index")
    
    # define model
    model = ResNet50()
    model.load_state_dict(
        torch.load(
            "/home/biopythoncodepc/Documents/git_repositories/Recommendation-Ranking-System/model_evaluation/1677054300/weights/10.pt",
            map_location=torch.device('cpu')))
    model.resnet50.fc = torch.nn.Linear(2048, 1000)
    

    with torch.no_grad():
        model.to(device)
        model.eval()
        feature = model(image_processor(pil_image).to(device))
        embedding = np.array(feature.cpu())
    distances, ann = index.search(embedding, k=index.ntotal)
    results = pd.DataFrame({"distances": distances[0], "ann": ann[0]})
    merge = pd.merge(results, reference, left_on="ann", right_index=True)
    labels = reference["id"]
    category = labels[ann[0][0]]
    print(category)

