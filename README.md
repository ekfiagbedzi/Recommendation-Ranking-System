# Recommendation Ranking System
## Description
This project involved the development and training of model that finds, recommends and ranks products, based on a user's search and previous interactions using Facebook's AI Similarity Search **(FAISS)**. Specifically, this project involved,
- Building a feature extraction model that generates embeddings from images and text data
- Generate a FAISS vector similarity search model to compare and select the most similar embeddings from different sets of embeddings
- Deploying both the Feature Extraction and FAISS models to an API which can serve requests from a potential customer
- Containerising the models and files using docker to ensure easier updating after retraining the models

## Installation
To install, run `docker pull emmacode/imclass:latest` on command line. You need to have docker installed to be able to do this.

## Usage
To use the application, run `docker run -it emmacode/imcalss:latest. Then visit http://0.0.0.0:8080 on you browser. You can then upload
an image and a text and hit run. The most similar image to your input will then be displayed

## File structure
.
├── LICENSE
├── README.md
├── app
│   ├── Dockerfile
│   ├── api.py
│   ├── docker-compose.yml
│   ├── image_decoder.json
│   ├── image_processor.py
│   └── requirements.txt
├── bin
│   ├── bert_model.py
│   ├── classifier.py
│   ├── custom_architecture.py
│   ├── faiss_model.py
│   ├── faiss_search.py
│   ├── featureX.py
│   ├── image_processor.py
│   ├── main.py
│   ├── regressor.py
│   ├── request.py
│   ├── resnet50.py
│   ├── setup.py
│   └── text_processor.py
├── data
│   ├── cleaned_images
│   └── tables
├── model_evaluation
│   ├── 1664455419
│   ├── 1664495576
│   ├── 1666666792
│   └── 1666858434
├── models
│   ├── combined_decoder.json
│   ├── image_decoder.json
│   └── text_decoder.json
├── requirements.txt
├── rout.md
├── testdata
│   └── dog.jpg
└── utils
    ├── __init__.py
    ├── clean_image_data.py
    ├── clean_tabular_data.ipynb
    ├── clean_tabular_data.py
    └── helpers.py


## Licence Information
MIT License

Copyright (c) 2022 Emmanuel Kwasi Fiagbedzi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
