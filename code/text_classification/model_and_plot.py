# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: model_and_plot
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
from keras.layers.core import Dropout
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, Bidirectional, Dense, Embedding, Dropout
from keras.models import Input, Model
from keras.optimizers import Adam

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
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64))(x)
    # Add a classifier
    outputs = Dense(1, activation="sigmoid")(x) # A sigmoid activation function as we are classifying binary data
    
    model = Model(inputs, outputs)
    model.summary() # Printing 
    model.compile(optimizer = "RMSprop", loss="binary_crossentropy", metrics=["accuracy"])
 
    history = model.fit(X_train, y_train, batch_size = batch_size, epochs=number_epochs, validation_data=(X_test, y_test))

    # Saving the model
    np.save('model_history/'+model_name+'.npy',history.history) 
    model.save("models/text_models/"+model_name+".h5")
    
    return history, model

##################################################
# FUNCTION TO PLOT THE HISTORY OF TRAINED MODELS #
##################################################
def plot_accuracy_loss_multiple(model_history_1, model_history_2, model_history_3, plt_name, loss=False):
    print("Saving model performance...")

    # Chaning between plot for accuracy or loss
    if loss == True:
        y_1 = 'loss'
        y_2 = 'val_loss'
    else:
        y_1 = 'acc'
        y_2 = 'val_acc'

    # Create figure.
    fig = plt.figure(figsize=(30, 10))
            
    # Setting values to rows and column variables
    rows = 1
    columns = 3

    # Adding suplot for Dataset 1
    fig.add_subplot(rows, columns, 1)
    plt.plot(model_history_1[y_1])
    plt.plot(model_history_1[y_2])
    plt.title('Dataset 1')
    plt.ylabel(y_1)
    plt.xlabel('epoch')
    plt.legend([y_1, y_2], loc='upper left')

    # Adding suplot for Dataset 2
    fig.add_subplot(rows, columns, 2)
    plt.plot(model_history_2[y_1])
    plt.plot(model_history_2[y_2])
    plt.title('Dataset 2')
    plt.ylabel(y_1)
    plt.xlabel('epoch')
    plt.legend([y_1, y_2], loc='upper left')

    # Adding suplot for Dataset 3
    fig.add_subplot(rows, columns, 3)
    plt.plot(model_history_3[y_1])
    plt.plot(model_history_3[y_2])
    plt.title('Dataset 3')
    plt.ylabel(y_1)
    plt.xlabel('epoch')
    plt.legend([y_1, y_2], loc='upper left')
        
    fig.savefig(plt_name+".jpg")



