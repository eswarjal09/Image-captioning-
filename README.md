# Image-captioning
This code implements a deep learning model which generates captions given an image.

## Architecture
![Image of Image captioning model](https://miro.medium.com/max/2000/1*ERwScS7k6IH3hZIJmGdHDg.png)

The model uses convolutional neural network(Transfer Learning) to extract features and a linear layer as an encoder. Bidirectional LSTM is used as decoder to extract text from the features extracted by pretrained CNN.

## Dataset
I am using Flickr dataset to train image captioning model which you could download from here. https://www.kaggle.com/hsankesara/flickr-image-dataset.
The data is split into train, validation and test and images are normalized and resized to a required size for the pretrained CNN model.
Using the captions for the training images we build vocabulary which we use to convert text to tokenids. We also add 'start' to the beginning of the caption and 'end' at the end of the caption. We also pad the text to a specific length for every batch so that we can train on a batch at the same time.


## Requirements
-pytorch - 1.0 
-nltk
-python - 3.6

## Training
python train.py 

## Sampling
python sample.py image_path
