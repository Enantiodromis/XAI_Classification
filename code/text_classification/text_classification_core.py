import re

import math
import keras
import keras.backend as backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow
from emot.emo_unicode import EMOTICONS, UNICODE_EMO
from keras.layers import Dense, Bidirectional, Embedding, LSTM
from keras.models import Input, Model, Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#######################################
# BUILD, TRAIN AND EVALUATE THE MODEL #
#######################################
def lstm_model(vocab_size, X_train, y_train, X_test, y_test, number_epochs, batch_size, model_name):
    print("Training model...")
   
    # Input shape is defined as shape(None,) for variable-length sequences of integers
    inputs = Input(shape=(None,), dtype="int32")
    # Each integer is embeded as a 128-dimensional vector
    x = Embedding(vocab_size, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)
    # Add a classifier
    outputs = Dense(1, activation="sigmoid")(x) # A sigmoid activation function as we are classifying binary data
    model = Model(inputs, outputs)
    model.summary() # Printing 
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
 
    history = model.fit(X_train, y_train, batch_size, epochs=number_epochs, validation_data=(X_test, y_test))

    # Saving the model
    np.save('model_history/'+model_name+'.npy',history.history)
    model.save("models/text_models/"+model_name+".h5")
    
    return history, model

####################################
# PLOTTING MODEL'S ACCURACY & LOSS #
####################################
def plot_accuracy_loss(model_history, number_epochs, model_name):
    print("Saving model performance...")
    x_list = []
    x_list.extend(range(0,number_epochs))

    plt.figure(1)
    plt.plot(model_history['accuracy'])
    plt.plot(model_history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks(x_list)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("data_plots/"+model_name+"_acc.jpg")

    plt.figure(2)
    plt.plot(model_history['loss'], color='green')
    plt.plot(model_history['val_loss'], color='red')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(x_list)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("data_plots/"+model_name+"_loss.jpg")


