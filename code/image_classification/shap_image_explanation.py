###########
# IMPORTS #
###########
from image_dataset_1_processing import get_dataset_1
from image_dataset_2_processing import get_dataset_2
from image_dataset_3_processing import get_dataset_3
import shap
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from model_and_plot import binary_dataset_creation, img_classification_model, plot_accuracy_loss
from keras.models import load_model 
import numpy as np
import pandas as pd
from PIL import ImageFile
import matplotlib.pyplot as plt
import random
from skimage import transform
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.applications import inception_v3 as inc_net

###############################
# EXTRACTING SHAP EXPLANATION #
###############################
def extracting_shap_explainer_explanation(train_generator, test_generator, model, save_path): 
    X_train, y_train = train_generator.next()
    X_test, y_test = test_generator.next()
    
    labels = list(train_generator.class_indices)
    background = X_train[np.random.choice(X_train.shape[0], 20, replace=False)]
    e = shap.DeepExplainer(model, background)
    
    test = X_test[:1]
    X_test_processed = [inc_net.preprocess_input(img) for img in test]
    shap_values = e.shap_values(X_test[:1])
    shap.image_plot(shap_values, X_test[:1], show=False)
    plt.savefig(save_path+'explanation_'+str(labels[y_test[1].astype(np.uint8)])+'_'+str(1)+'_.png')
    plt.close() 
    
    for el in range(len(test)):
        transformed_img = transform.rotate(X_test_processed[el], angle=-20, cval=205)
        pred_peturb = model.predict(np.expand_dims(transformed_img,axis=0))
        plt.imshow(transformed_img)
        plt.savefig(save_path+'explanation_'+str(pred_peturb)+'_'+str(el)+'_peturb.png')
        plt.close()
        

############################################################
# INITIALISING MODEL & DATA FOR FAKE VS REAL FACES DATASET #
############################################################
train_generator, test_generator, valid_generator = get_dataset_1()
model_name = "shap_xai_image_classification_data_1_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/shap_image_explanations/image_data_1/'
extracting_shap_explainer_explanation(train_generator, test_generator, model, save_path)

"""####################################################
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
extracting_shap_explainer_explanation(train_generator, valid_generator, model, save_path)"""


