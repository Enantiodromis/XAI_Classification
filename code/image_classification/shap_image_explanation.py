###########
# IMPORTS #
###########
from dataset_processing.image_dataset_1_processing import get_dataset_1
from dataset_processing.image_dataset_2_processing import get_dataset_2
from dataset_processing.image_dataset_3_processing import get_dataset_3
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
def extracting_shap_explainer_explanation(train_generator, test_generator, model, save_path): 
    X_train, y_train = train_generator.next()
    X_test, y_test = test_generator.next()

    labels = list(train_generator.class_indices)
    random_indexes = random.sample(range(1,len(X_test)),3)
    background = X_train[np.random.choice(X_train.shape[0], 20, replace=False)]
    e = shap.DeepExplainer(model, background)

    for x in range(3):
        for index in random_indexes:
            shap_values = e.shap_values(X_test[index-1:index])
            shap.image_plot(shap_values, X_test[index-1:index], show=False)
            plt.savefig(save_path+'explanation_'+str(labels[y_test[index].astype(np.uint8)])+'_'+str(index)+'_.png')
            plt.close()

############################################################
# INITIALISING MODEL & DATA FOR FAKE VS REAL FACES DATASET #
############################################################
train_generator, test_generator, valid_generator = get_dataset_1()
model_name = "shap_xai_image_classification_data_1_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/shap_image_explanations/image_data_1/'
extracting_shap_explainer_explanation(train_generator, test_generator, model, save_path)

####################################################
# INITIALISING MODEL & DATA FOR CAT VS DOG DATASET #
####################################################
train_generator, test_generator = get_dataset_2()
model_name = "shap_xai_image_classification_data_2_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/shap_image_explanations/image_data_2/'
extracting_shap_explainer_explanation(train_generator, test_generator, model, save_path)

###########################################################
# INITIALISING MODEL & DATA FOR WATERMARK VS NO_WATERMARK #
###########################################################
train_generator, valid_generator = get_dataset_3()
model_name = "shap_xai_image_classification_data_3_ConvNet"
model = load_model("models/image_models/"+model_name+".h5")
save_path = 'image_explanations/shap_image_explanations/image_data_3/'
extracting_shap_explainer_explanation(train_generator, valid_generator, model, save_path)


