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
import image_processor
import torch.nn.functional as F


with open("image_decoder.json", "r") as f:
    image_decoder = json.load(f)
##############################################################
# TODO                                                       #
# Import your image and text processors here                 #
##############################################################


class ImageClassifier(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(ImageClassifier, self).__init__()
        self.resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_resnet50',
            pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13)

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the image model   #
##############################################################
        
        self.decoder = decoder

    def forward(self, image):
        return F.softmax(self.resnet50(image), dim=1)

    def predict(self, image):
        with torch.no_grad():
            predictions = self.forward(image)
            return self.decoder[str(torch.argmax(predictions).item())], predictions.tolist()

    def predict_proba(self, image):
        with torch.no_grad():
            return torch.max(self.predict(image))

    def predict_classes(self, image):
        with torch.no_grad():
            return self.decoder[str(torch.argmax(self.predict(image)).item())]


# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str


try:
##############################################################
# TODO                                                       #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the image model   #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# image_decoder.pkl                                          #
##############################################################
    model = ImageClassifier(image_decoder)
    model.load_state_dict(torch.load("2.pt", map_location=torch.device("cpu")))
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")


try:
##############################################################
# TODO                                                       #
# Initialize the image processor that you will use to process#
# the text that you users will send to your API              #
##############################################################
    image_processor
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
    category, probs = model.predict(image_processor.image_processor(pil_image))
    return JSONResponse(content={
    "Category": "{}".format(category), # Return the category here
    "Probabilities": "{}".format(probs) # Return a list or dict of probabilities here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)