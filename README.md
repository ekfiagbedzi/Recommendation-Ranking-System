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

## Licence Information

