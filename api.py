import pickle
import json
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from image_processor import process_image


class ImageClassifier(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_resnet50',
            pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13)        
        self.decoder = decoder

    def forward(self, image):
        x = self.main(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_proba(self, image):
        with torch.no_grad():
            pass

    def predict_classes(self, image):
        with torch.no_grad():
            pass


try:

    image_model = ImageClassifier()
    image_model.load_state_dict(
        torch.load(
            "/home/biopythoncodepc/Documents/Recommendation-Ranking-System/model_evaluation/1677054215/weights/8.pt"))

    with open("image_decoder.json", "r") as f:
        image_decoder = json.load(f)

except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
    image_processor = process_image("0a1baaa8-4556-4e07-a486-599c05cce76c")
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}


@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
    "Category": "", # Return the category here
    "Probabilities": "" # Return a list or dict of probabilities here
        })
  
@app.post('/predict/indices')
def predict_indices(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    
    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # In this case, text is the text that the user sent to your  #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
    "Category": "", # Return the category here
    "Probabilities": "" # Return a list or dict of probabilities here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)