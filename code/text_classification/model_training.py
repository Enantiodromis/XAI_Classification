# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: model_training
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
import numpy as np

from model_and_plot import lstm_model, plot_accuracy_loss_multiple
from text_dataset_1_processing import get_dataset_1
from text_dataset_2_processing import get_dataset_2
from text_dataset_3_processing import get_dataset_3

#####################################
# TRAINING MODEL FOR TEXT_DATASET 1 #
#####################################
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_1(0.2) # Processing the data preparing for training
model_name = "xai_text_classification_data_1_lstm" # Name of associated model
#history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, 5, 16, model_name) # Training the models. 
history_1=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting

#####################################
# TRAINING MODEL FOR TEXT_DATASET 2 #
#####################################
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_2(0.2) # Processing the data preparing for training
model_name = "xai_text_classification_data_2_lstm" # Name of associated model
#history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, 5, 16, model_name)
history_2=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting

#####################################
# TRAINING MODEL FOR TEXT_DATASET 3 #
#####################################
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_3(0.2, 80000) # Processing the data preparing for training
model_name = "xai_text_classification_data_3_lstm" # Name of associated model
#history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, 5, 16, model_name)
history_3=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting


# Plotting the accuracy and loss of the model on each of the datasets
plot_accuracy_loss_multiple(history_1, history_2, history_3, "data_plots/acc_plot_text_data", False)
plot_accuracy_loss_multiple(history_1, history_2, history_3, "data_plots/loss_plot_text_data", True)
