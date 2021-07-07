###########
# IMPORTS #
###########
import numpy as np 
import os 
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from tensorflow.python.ops.gen_array_ops import tensor_scatter_add_eager_fallback

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

def dataset_creation(batch_size, img_height, img_width, dataframe):
    data_generator = ImageDataGenerator()
    generator = data_generator.flow_from_dataframe(
        dataframe = dataframe,
        x_col= 'path',
        y_col= 'label_str',
        batch_size= batch_size,
        target_size= (img_height,img_width),
        color_mode= 'rgb',
        class_mode= 'categorical'
    )

test_df = pd.read_csv('datasets/image_data_1/test.csv')
train_df = pd.read_csv('datasets/image_data_1/train.csv')
valid_df = pd.read_csv('datasets/image_data_1/valid.csv')

data_details("datasets/image_data_1/real_vs_fake")
dataset_creation(32, 256, 256, test_df)
dataset_creation(32, 256, 256, train_df)
dataset_creation(32, 256, 256, valid_df)