import json
import numpy as np
import faiss


with open("image_embeddings.json", "r") as f:
    image_embeddings = json.load(f)

embeddings = np.asarray(list(image_embeddings.values())).astype("float32")
print(embeddings.shape)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(index.ntotal)
D, I = index.search(embeddings, k=4)
print(I)
