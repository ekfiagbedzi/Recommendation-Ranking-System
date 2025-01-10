import json
import numpy as np
import faiss
from faiss import write_index


with open("image_embeddings.json", "r") as f:
    image_embeddings = json.load(f)

embeddings = np.asarray(list(image_embeddings.values())).astype("float32")
print(embeddings.shape)
index = faiss.IndexFlatL2(embeddings.shape[1])
faiss.normalize_L2(embeddings)
index.add(embeddings)
write_index(index, "rrs.index")