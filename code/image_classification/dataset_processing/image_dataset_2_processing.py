from image_classification_core import binary_dataset_creation
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset_2():
    folder_path = 'datasets/image_data/image_data_2'
    train_generator, test_generator = binary_dataset_creation(32, 256, 256, False, True, file_path=folder_path)
    return train_generator, test_generator