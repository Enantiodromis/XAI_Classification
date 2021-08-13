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
from PIL import ImageFile
import matplotlib.pyplot as plt
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

###############################
# EXTRACTING SHAP EXPLANATION #
###############################
def extracting_shap_explainer_explanation(train_generator, test_generator, model): 
    X_train, y_train = train_generator.next()
    X_test, y_test = test_generator.next()

    labels = list(train_generator.class_indices)
    print(labels)
    random_indexes = random.sample(range(1,len(X_test)),3)
    background = X_train[np.random.choice(X_train.shape[0], 20, replace=False)]
    e = shap.DeepExplainer(model, background)

    for x in range(3):
        for index in random_indexes:
            shap_values = e.shap_values(X_test[index-1:index])
            shap.image_plot(shap_values, X_test[index-1:index], show=False)
            plt.savefig('image_explanations/shap_image_explanations/explanation_'+str(labels[y_test[index].astype(np.uint8)])+'_'+str(index)+'_.png')
            plt.close()

############################################################
# INITIALISING MODEL & DATA FOR FAKE VS REAL FACES DATASET #
############################################################
test_df = pd.read_csv('datasets/image_data/image_data_1/test.csv')
train_df = pd.read_csv('datasets/image_data/image_data_1/train.csv')
test_generator = binary_dataset_creation(32, 256, 256, True, False, dataframe=test_df)
train_generator = binary_dataset_creation(32, 256, 256, True, False, dataframe=train_df)

model_name = "shap_xai_image_classification_data_1_ConvNet" # Initialising a model name
#history, model = img_classification_model(train_generator, test_generator, 60, model_name) # Training the model
history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
fake_real_faces_classification_model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
#plot_accuracy_loss(history, 20, model_name) # Plotting the training history and saving the plots
extracting_shap_explainer_explanation(train_generator, test_generator, fake_real_faces_classification_model)

####################################################
# INITIALISING MODEL & DATA FOR CAT VS DOG DATASET #
####################################################
folder_path = 'datasets/image_data/image_data_2'
# Creating the necessary image generators to train and test the model
train_generator, test_generator = binary_dataset_creation(32, 256, 256, False, True, file_path=folder_path) 
model_name = "shap_xai_image_classification_data_2_ConvNet" # Initialising a model name
#history, model = img_classification_model(train_generator, test_generator, 60, model_name) # Training the model
history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
cat_dog_classification_model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
#plot_accuracy_loss(history, 20, model_name) # Plotting the training history and saving the plots
extracting_shap_explainer_explanation(train_generator, test_generator, cat_dog_classification_model)

###########################################################
# INITIALISING MODEL & DATA FOR WATERMARK VS NO_WATERMARK #
###########################################################
folder_path_train = 'datasets/image_data/image_data_3/train'
folder_path_valid = 'datasets/image_data/image_data_3/valid'


train_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_train)
valid_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_valid)

X_test, y_test = next(valid_generator)

model_name = "shap_xai_image_classification_data_3_ConvNet"
#history, model = img_classification_model(train_generator, valid_generator, 60, model_name)
model = load_model("models/image_models/"+model_name+".h5")

extracting_shap_explainer_explanation(train_generator, valid_generator, model)
#history=np.load('model_history/image_classification_ConvNet_image_data_1.npy',allow_pickle='TRUE').item()
#plot_accuracy_loss(history, 50)


