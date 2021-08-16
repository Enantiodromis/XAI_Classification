from image_classification_core import binary_dataset_creation
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset_3():
    folder_path_train = 'datasets/image_data/image_data_3/train'
    folder_path_valid = 'datasets/image_data/image_data_3/valid'

    train_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_train)
    valid_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_valid)

    return train_generator, valid_generator