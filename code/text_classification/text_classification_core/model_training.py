import numpy as np

from dataset_processing.text_dataset_1_processing import get_dataset_1
from dataset_processing.text_dataset_2_processing import get_dataset_2
from dataset_processing.text_dataset_3_processing import get_dataset_3
from model_and_plot import lstm_model, plot_accuracy_loss

#####################################
# TRAINING MODEL FOR TEXT_DATASET 1 #
#####################################
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_1(0.33) # Processing the data preparing for training
model_name = "xai_text_classification_data_1_lstm" # Name of associated model
history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, 2, 32, model_name) # Training the models. 
#history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
#plot_accuracy_loss(history, epoch_num, model_name)

#####################################
# TRAINING MODEL FOR TEXT_DATASET 2 #
#####################################
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_2(0.33) # Processing the data preparing for training
model_name = "xai_text_classification_data_2_lstm" # Name of associated model
history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, 2, 32, model_name)
#history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
#plot_accuracy_loss(history, epoch_num, model_name)

#####################################
# TRAINING MODEL FOR TEXT_DATASET 3 #
#####################################
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_3(0.33, 40000) # Processing the data preparing for training
model_name = "xai_text_classification_data_3_lstm" # Name of associated model
history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, 2, 32, model_name)
#history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
#plot_accuracy_loss(history, epoch_num, model_name)
