# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: image_dataset_2_processing
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
from PIL import ImageFile
from model_and_plot import binary_dataset_creation

ImageFile.LOAD_TRUNCATED_IMAGES = True

#######################
# PREPARING DATASET 2 #
#######################
def get_dataset_2():
    folder_path = 'datasets/image_data/image_data_2'
    train_generator, test_generator = binary_dataset_creation(16, 256, 256, False, True, file_path=folder_path)
    return train_generator, test_generator
