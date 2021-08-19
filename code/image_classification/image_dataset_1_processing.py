from model_and_plot import binary_dataset_creation
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset_1():
    folder_path_train = 'datasets/image_data/image_data_1/real_vs_fake/train'
    folder_path_test = 'datasets/image_data/image_data_1/real_vs_fake/test'
    folder_path_valid = 'datasets/image_data/image_data_1/real_vs_fake/valid'

    train_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_train)
    test_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_test)
    valid_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_valid)

    return train_generator, test_generator, valid_generator
