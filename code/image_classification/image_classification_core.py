###########
# IMPORTS #
###########
import json
import os
import pathlib
import pickle
import random
from time import time

import numpy as np
import pandas as pd
import PIL
import PIL.Image
import tensorflow as tf
from keras.applications import inception_v3 as inc_net
from keras.backend import exp
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPool2D, MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import ravel
from scipy.ndimage import interpolation
from skimage import segmentation
from skimage.color import label2rgb
from skimage.color.colorconv import gray2rgb, rgb2gray
from sklearn.utils import shuffle
from keras.models import load_model
from keras.optimizers import RMSprop


################
# DATA DETAILS #
################
def data_details(data_path):
    data_dir = pathlib.Path(data_path)
    tt_list = ["train", "test", "valid"]
    for test_or_train in tt_list:
        if test_or_train == "train":
            print("#########################")
            print("# Training data details #")
            print("#########################")
        if test_or_train == "test":
            print("########################")
            print("# Testing data details #")
            print("########################")
        if test_or_train == "valid":
            print("###########################")
            print("# Validation data details #")
            print("###########################")

        real_path = test_or_train+'/real/*.jpg'
        image_count_real = len(list(data_dir.glob(real_path)))
        print("Number of real images in the %s dataset:" % test_or_train, image_count_real)
        
        fake_path = test_or_train+'/fake/*.jpg'
        image_count_fake = len(list(data_dir.glob(fake_path)))
        print("Number of fake images in the %s dataset:" % test_or_train, image_count_fake)
        print("Total number of images:", image_count_real+image_count_fake)
        print("")

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
        data_generator = ImageDataGenerator(validation_split=0.2)
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
def img_classification_model(train_generator, test_generator, number_epochs, files_name):
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
                optimizer=RMSprop(lr=1e-4),
                metrics=['acc'])

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator) ,
        epochs=number_epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
    )
    np.save('model_history/'+files_name+'.npy',history.history)
    model.save('models/'+files_name+'.h5') 

    return history, model

def plot_accuracy_loss(model_history, number_epochs, model_name):
    x_list = []
    x_list.extend(range(number_epochs))

    plt.figure(2,figsize=(15,4))
    plt.plot(model_history['acc'])
    plt.plot(model_history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks(x_list)
    plt.tight_layout()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("data_plots/"+model_name+"_acc.jpg")

    plt.figure(3,figsize=(15,4))
    plt.plot(model_history['loss'], color='green')
    plt.plot(model_history['val_loss'], color='red')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(x_list)
    plt.tight_layout()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("data_plots/"+model_name+"_loss.jpg")

"""
x_test, y_test = next(valid_generator)
path_list_2 = []
for file in valid_generator.filenames:
    path_list_2.append("datasets\\image_data_2\\"+file)

labels_2 = list(valid_generator.class_indices.keys())
"""

"""import os
from PIL import Image
folder_path = 'datasets/image_data_2'
extensions = []
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        print('** Path: {}  **'.format(file_path), end="\r", flush=True)
        im = Image.open(file_path)
        rgb_im = im.convert('RGB')
        if filee.split('.')[1] not in extensions:
            extensions.append(filee.split('.')[1])"""

