###########
# IMPORTS #
###########
import shap
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from image_classification_core import binary_dataset_creation, img_classification_model, plot_accuracy_loss
from keras.models import load_model 
import numpy as np
import pandas as pd

###############################
# EXTRACTING SHAP EXPLANATION #
###############################
def extracting_shap_explainer_explanation(train_generator, test_generator, model): 
    X_train, y_train = train_generator.next()
    X_test, y_test = test_generator.next()

    background = X_train[np.random.choice(X_train.shape[0], 10, replace=False)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(X_test[:1])
    shap.image_plot(shap_values, X_test[:1])

############################################################
# INITIALISING MODEL & DATA FOR FAKE VS REAL FACES DATASET #
############################################################
test_df = pd.read_csv('datasets/image_data_1/test.csv')
train_df = pd.read_csv('datasets/image_data_1/train.csv')
test_generator = binary_dataset_creation(32, 256, 256, True, False, dataframe=test_df)
train_generator = binary_dataset_creation(32, 256, 256, True, False, dataframe=train_df)

model_name = "shap_xai_image_classification_ConvNet_fake_real_faces" # Initialising a model name
history, model = img_classification_model(train_generator, test_generator, 5, model_name) # Training the model
#history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
fake_real_faces_classification_model = load_model("models/"+model_name+".h5") # Loading the saved model
#plot_accuracy_loss(history, 20, model_name) # Plotting the training history and saving the plots
extracting_shap_explainer_explanation(train_generator, test_generator, fake_real_faces_classification_model)

####################################################
# INITIALISING MODEL & DATA FOR CAT VS DOG DATASET #
####################################################
folder_path = 'datasets/image_data_2'
# Creating the necessary image generators to train and test the model
train_generator, test_generator = binary_dataset_creation(32, 256, 256, False, True, file_path=folder_path) 
model_name = "shap_xai_image_classification_ConvNet_cat_dog" # Initialising a model name
#history, model = img_classification_model(train_generator, test_generator, 20, model_name) # Training the model
#history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
cat_dog_classification_model = load_model("models/"+model_name+".h5") # Loading the saved model
#plot_accuracy_loss(history, 20, model_name) # Plotting the training history and saving the plots
extracting_shap_explainer_explanation(train_generator, test_generator, cat_dog_classification_model)

