# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: model_and_plot
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
import numpy as np
import tensorflow.keras
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

####################
# DATASET CREATION #
####################
def binary_dataset_creation(batch_size, img_height, img_width, from_dataframe, need_train_test_split, dataframe = None, file_path = None):
    if need_train_test_split == False:
        data_generator = ImageDataGenerator()
        if from_dataframe == True:
            generator = data_generator.flow_from_dataframe(
                dataframe = dataframe,
                x_col= 'path',
                y_col= 'label_str',
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                shuffle=True
            )
        else:
            generator = data_generator.flow_from_directory(
                directory = file_path,
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                shuffle=True   
            )
        return generator
    else: 
        data_generator = ImageDataGenerator(validation_split=0.33)
        if from_dataframe == True:
            generator_1 = data_generator.flow_from_dataframe(
                dataframe = dataframe,
                x_col= 'path',
                y_col= 'label_str',
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="training",
                shuffle=True
            )
            generator_2 = data_generator.flow_from_dataframe(
                dataframe = dataframe,
                x_col= 'path',
                y_col= 'label_str',
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="validation",
                shuffle=True
            )
        else:
            generator_1 = data_generator.flow_from_directory(
                directory = file_path,
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="training",
                shuffle=True   
            )
            generator_2 = data_generator.flow_from_directory(
                directory = file_path,
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="validation",
                shuffle=True   
            )
        return generator_1, generator_2

#######################################
# BUILD, TRAIN AND EVALUATE THE MODEL #
#######################################
def img_classification_model(train_generator, test_generator, number_epochs, model_name):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer= RMSprop(lr=1e-4),
                metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=number_epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
    )
    np.save('model_history/'+model_name+'.npy',history.history)
    model.save('models/image_models/'+model_name+'.h5') 

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
        y_1 = 'accuracy'
        y_2 = 'val_accuracy'

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
