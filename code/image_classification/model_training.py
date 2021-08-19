from model_and_plot import img_classification_model

from image_dataset_1_processing import get_dataset_1
from image_dataset_2_processing import get_dataset_2
from image_dataset_3_processing import get_dataset_3

#####################################
# TRAINING MODEL FOR TEXT_DATASET 1 #
#####################################
train_generator, test_generator, valid_generator = get_dataset_1()
model_name = "shap_image_classification_data_1_ConvNet"
history, model = img_classification_model(train_generator, test_generator, 2, model_name)

"""#####################################
# TRAINING MODEL FOR TEXT_DATASET 2 #
#####################################
train_generator, test_generator = get_dataset_2()
model_name = "xai_image_classification_data_2_ConvNet_anchor" 
history, model = img_classification_model(train_generator, test_generator, 60, model_name)

#####################################
# TRAINING MODEL FOR TEXT_DATASET 3 #
#####################################
train_generator, valid_generator = get_dataset_3()
model_name = "xai_image_classification_data_3_ConvNet"
history, model = img_classification_model(train_generator, test_generator, 60, model_name)"""

