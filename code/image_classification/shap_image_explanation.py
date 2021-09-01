# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: shap_image_explanation
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
import random

import matplotlib.pyplot as plt
import numpy as np
import shap
from skimage import transform
from tensorflow.keras import backend
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.models import load_model

from image_dataset_1_processing import get_dataset_1
from image_dataset_2_processing import get_dataset_2
from image_dataset_3_processing import get_dataset_3

from image_perturbation import image_perturbation

###############################
# EXTRACTING SHAP EXPLANATION #
###############################
def extracting_shap_explanation(train_generator, test_generator, model, save_path, perturb=False): 
    
    # Getting x and y data from image generator
    X_train, y_train = train_generator.next()
    X_test, y_test = test_generator.next()
    
    # Extracting the class names from the dataset.
    classes = list(train_generator.class_indices)
    
    # Creating a background for the explainer and initialising the explainer.
    background = X_train[np.random.choice(X_train.shape[0], 16, replace=False)]
    e = shap.DeepExplainer(model, background)

    num_instances_explain = 10 # <--- Change this value to alter the amount of generated explanations.

    # Iterating over the defined number of explanations wanted.
    for instance in range(num_instances_explain):
        random_index = random.randint(1,len(X_test)) # Generating a random number in the range of the length of X_test.
        pred = model.predict(np.expand_dims(X_test[random_index],axis=0)) # Calculating the prediction for the original image.
        shap_values = e.shap_values(X_test[random_index:random_index+1]) # Calculating the shap values for the image which we are explaining.
        shap.image_plot(shap_values, X_test[random_index:random_index+1], show=False) # Plotting the explanation.
        plt.savefig(save_path+'explanation_'+str(instance)+'_'+str(classes[round(pred[0][0])])+'.png') # Saving the explanation.
        plt.close() 
        
        # Implementation to handle perturbed data, due to shap's implementation we are unable to save within one plot, so the perturbed data is plotted separately.
        if perturb == True: 
            X_test_processed = inc_net.preprocess_input(X_test[random_index]) # Processing the raw image data using the inception_v3 model.
            perturbed_img = image_perturbation(X_test_processed)
            pred_peturb = model.predict(np.expand_dims(perturbed_img,axis=0)) # Predicitng classification for the perturbed data.
            plt.title('Peturbed') # Adding a title to the perturbed data plot
            plt.imshow(perturbed_img) # Plotting the perturbed data
            plt.savefig(save_path+'explanation_'+str(instance)+'_perturbed_'+str(classes[round(pred_peturb[0][0])])+'.png') # Saving the perturbed data
            plt.close()
    
############################################################
# INITIALISING MODEL & DATA FOR FAKE VS REAL FACES DATASET #
############################################################
train_generator_1, test_generator_1, valid_generator_1 = get_dataset_1()
model_name = "xai_image_classification_data_1_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/shap_image_explanations/image_data_1/'
extracting_shap_explanation(train_generator_1, test_generator_1, model, save_path, perturb=True)

####################################################
# INITIALISING MODEL & DATA FOR CAT VS DOG DATASET #
####################################################
train_generator_2, test_generator_2 = get_dataset_2()
model_name = "xai_image_classification_data_2_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/shap_image_explanations/image_data_2/'
extracting_shap_explanation(train_generator_2, test_generator_2, model, save_path, perturb=True)

###########################################################
# INITIALISING MODEL & DATA FOR WATERMARK VS NO_WATERMARK #
###########################################################
train_generator_3, test_generator_3 = get_dataset_3()
model_name = "xai_image_classification_data_3_ConvNet"
model = load_model("models/image_models/"+model_name+".h5")
save_path = 'image_explanations/shap_image_explanations/image_data_3/'
extracting_shap_explanation(train_generator_3, test_generator_3, model, save_path, perturb=True)


