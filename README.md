# Cats and Dogs Classification using CNN and Streamlit

This project implements a **Convolutional Neural Network (CNN)** to classify images of cats and dogs. The model is integrated into a **Streamlit** web application, where users can upload an image, and the model will predict whether the image is of a cat or a dog.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
- [Results](#results)

## Project Overview
This project classifies images of cats and dogs using a CNN built with Keras and TensorFlow. The model is deployed in a Streamlit app where users can upload their own images for prediction.

## Dataset
The dataset used in this project consists of images of cats and dogs. The model was trained using the following data:
- **Training Set**: A collection of labeled images of cats and dogs.
- **Testing Set**: A separate collection for model evaluation.

You can download the dataset from [Kaggle - Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data).

## Model Architecture
The model is a simple CNN with the following layers:
- **Convolutional Layers**: Extract features from input images.
- **MaxPooling Layers**: Reduce the dimensionality of the features.
- **Flatten Layer**: Convert 2D features to 1D.
- **Dense Layers**: Fully connected layers for classification.

The output is a binary classification (0 for cat, 1 for dog).

## Requirements

### Key Libraries:
+ **Streamlit:** For creating the interactive web app.
+ **TensorFlow:** For loading and running the pre-trained CNN model.
+ **Keras:** For building and training the CNN model.
+ **NumPy:** For numerical operations on image data.

## How It Works
+ Upload an Image: The user uploads an image of either a cat or a dog.
+ Image Preprocessing: The uploaded image is resized to 64x64 pixels to match the input dimensions of the CNN.
+ Prediction: The pre-trained CNN model predicts the class (cat or dog) of the uploaded image.
+ Display Result: The result (Cat or Dog) is shown on the screen.

### Example Screenshot
![image](https://github.com/user-attachments/assets/b30c6fd3-0b26-40bb-9253-95e91c09bb21)
![image](https://github.com/user-attachments/assets/712bbc95-70a4-4996-a7c7-47d2b095a680)


Streamlit Website [LINK](https://dogs-vs-cats-using-cnn.streamlit.app/)

## Results
The model was trained on 8,000 images (4000 cats and 4000 dogs) and achieved an **accuracy of ~91%** on the test set.

## Author
Siddharth Jaiswal
