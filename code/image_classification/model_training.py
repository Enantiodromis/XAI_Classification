# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: model_training
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
import numpy as np
from keras.models import load_model

from image_dataset_1_processing import get_dataset_1
from image_dataset_2_processing import get_dataset_2
from image_dataset_3_processing import get_dataset_3

from model_and_plot import img_classification_model, plot_accuracy_loss_multiple

#####################################
# TRAINING MODEL FOR TEXT_DATASET 1 #
#####################################
train_generator_1, test_generator_1, valid_generator_1 = get_dataset_1() # Getting the prepared data
model_name = "xai_image_classification_data_1_ConvNet" # Defining a model name
history, model = img_classification_model(train_generator_1, test_generator_1, 5, model_name) # Training the model
history_1=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
model = load_model("models/image_models/"+model_name+".h5") # Loading the trained model.

#####################################
# TRAINING MODEL FOR TEXT_DATASET 2 #
#####################################
train_generator_2, test_generator_2 = get_dataset_2() # Getting the prepared data
model_name = "xai_image_classification_data_2_ConvNet" # Defining a model name
history, model = img_classification_model(train_generator_2, test_generator_2, 5, model_name)
history_2=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
model = load_model("models/image_models/"+model_name+".h5") # Loading the trained model.

#####################################
# TRAINING MODEL FOR TEXT_DATASET 3 #
#####################################
train_generator_3, valid_generator_3 = get_dataset_3() # Getting the prepared data
model_name = "xai_image_classification_data_3_ConvNet" # Defining a model name
history, model = img_classification_model(train_generator_3, valid_generator_3, 5, model_name) # Training the model
history_3=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
model = load_model("models/image_models/"+model_name+".h5") # Loading the trained model.

# Plotting the accuracy and loss of the model on each of the datasets
plot_accuracy_loss_multiple(history_1, history_2, history_3, "data_plots/acc_plot_image_data", False)
plot_accuracy_loss_multiple(history_1, history_2, history_3, "data_plots/loss_plot_image_data", True)
