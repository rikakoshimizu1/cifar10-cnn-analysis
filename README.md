# CIFAR-10 CNN Classification

## Overview
This project involves building and improving a **simple Convolutional Neural Network (CNN)** model for image classification using the **CIFAR-10 dataset**. The dataset contains **60,000 32x32 color images** categorized into **10 distinct classes**. The goal is to design, train, and evaluate CNN models while experimenting with different architectures and techniques to improve performance. 

## Objective
The objective of this assignment is to develop a simple CNN and enhance it through additional **convolutional layers**, **max pooling layers**, and **dropout regularization**. The models are trained and validated to observe the effects of these modifications on **accuracy** and **overfitting**.

## Dataset
The **CIFAR-10 dataset** consists of **60,000 32x32 pixel color images** distributed across **10 classes** with **6,000 images per class**. It is divided into:
 - **50,000 training images**
 - **10,000 test images**

The training data is further split into five batches of 10,000 images each. The test batch contains 1,000 randomly selected images from each class.

## Classes
 - `Airplane`
 - `Automobile`
 - `Bird`
 - `Cat`
 - `Deer`
 - `Dog`
 - `Frog`
 - `Horse`
 - `Ship`
 - `Truck`

## Features
 - Images are **RGB colored** and `32x32` pixels in size.
 - Labels are **categorical values** corresponding to the 10 classes.
 - Preprocessing includes **normalization** and **one-hot encoding** of labels.
 - Dataset is split into **training**, **validation** (20% of training data), and **test** sets.

## Challenges
 - **Overfitting**: Initial models showed signs of overfitting, where **training accuracy** improved while **validation accuracy** plateaued or worsened.
 - **Model Design**: Determining the optimal number of **convolutional** and **pooling layers** was challenging to avoid underfitting or overfitting.
 - **Hyperparameter Tuning**: Selecting suitable **dropout rates** and layer sizes required experimentation for improved generalization.
 - **Class Imbalance in Batches**: Random distribution of images in training batches sometimes resulted in uneven class representation, affecting learning stability.

## Analysis and Insights
 - Visualization of sample images confirmed correct dataset loading and preprocessing. **Normalization** of pixel values helped with model convergence.
 - The initial CNN model provided baseline performance. Adding more **convolutional** and **max pooling layers** helped extract more features but risked overfitting.
 - Incorporation of **dropout layers** in the third model effectively mitigated overfitting, as indicated by training loss plateauing and validation accuracy aligning more closely with training accuracy.
 - **Performance Comparison**
   - Models 1 and 2 showed decreasing training loss but diverging validation loss and accuracy, confirming **overfitting issues**.
   - Model 3's **dropout layers** improved validation metrics, highlighting the importance of **regularization** in CNNs.
 - Further tuning of **hyperparameters**, experimenting with **dense layers**, adjusting **dropout rates**, and possibly incorporating **batch normalization** could enhance model performance.

## About
This Jupyter Notebook was completed as part of the **UCSC Silicon Valley Extension Deep Learning and Artificial Intelligence** program.
