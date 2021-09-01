# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: image_dataset_3_processing
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
from PIL import ImageFile
from model_and_plot import binary_dataset_creation

ImageFile.LOAD_TRUNCATED_IMAGES = True

#######################
# PREPARING DATASET 3 #
#######################
def get_dataset_3():
    folder_path_train = 'datasets/image_data/image_data_3/train'
    folder_path_valid = 'datasets/image_data/image_data_3/valid'

    train_generator = binary_dataset_creation(16, 256, 256, False, False, file_path=folder_path_train)
    valid_generator = binary_dataset_creation(16, 256, 256, False, False, file_path=folder_path_valid)

    return train_generator, valid_generator
