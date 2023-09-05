# Sentiment_LSTM
Title: Sentiment Analysis Using LSTM

## Overview
This repository contains Python code for sentiment analysis using a Long Short-Term Memory (LSTM) neural network. Sentiment analysis is the process of determining the sentiment or emotional tone of a piece of text, such as a movie review. In this project, we use TensorFlow and Keras to build and train an LSTM-based sentiment analysis model.

## Dependencies
- NumPy: A library for numerical operations in Python.
- TensorFlow and Keras: For building and training the deep learning model.
- Scikit-learn: For data preprocessing and evaluation.

## Sentiment Analysis Model
The sentiment analysis model consists of the following components:

1. **Tokenizer**: We use the Keras `Tokenizer` to preprocess and tokenize the text data. It converts text reviews into sequences of integers, where each integer corresponds to a word in the vocabulary.

2. **Embedding Layer**: This layer converts the integer sequences into dense vector representations. It helps the model understand the meaning of words and their context.

3. **Bidirectional LSTM**: The LSTM (Long Short-Term Memory) layer is a type of recurrent neural network (RNN) that is well-suited for sequence data. We use a bidirectional LSTM to capture information from both directions in the sequence, enhancing the model's ability to understand context.

4. **Dense Layer**: The final dense layer with a sigmoid activation function produces a binary sentiment classification output (positive or negative).

## Data
We provide a small dataset of movie reviews along with their corresponding labels (1 for positive and 0 for negative).

## Training and Evaluation
The code includes the following steps:

1. Data preprocessing: Tokenization and padding of sequences.
2. Splitting the data into training and validation sets.
3. Model creation and compilation.
4. Model training for sentiment analysis.
5. Evaluation of the model's performance on the validation set, including accuracy.

## Usage
1. Install the required dependencies: NumPy, TensorFlow, Keras, and Scikit-learn.
2. Run the provided code to train the sentiment analysis model on the provided dataset.
3. The model's validation accuracy will be displayed in the console.

You can also modify the code to use your own dataset for sentiment analysis or experiment with different hyperparameters to achieve better results.

