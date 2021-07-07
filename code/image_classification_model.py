###########
# IMPORTS #
###########
import numpy as np 
import os 
import PIL
import PIL.Image
import tensorflow as tf
import pathlib

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

data_details("datasets/image_data_1/real_vs_fake")
