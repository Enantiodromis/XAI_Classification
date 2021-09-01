# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: lime_image_explanation
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
import random
from re import X

import numpy as np
from keras.applications import inception_v3 as inc_net
from keras.models import load_model
from lime import lime_image
from matplotlib import pyplot as plt
from skimage import transform
from skimage.segmentation import mark_boundaries

from image_dataset_1_processing import get_dataset_1
from image_dataset_2_processing import get_dataset_2
from image_dataset_3_processing import get_dataset_3

from image_perturbation import image_perturbation

###################################
# LIME IMAGE EXPLANATION FUNCTION #
###################################
def extracting_lime_explanation(model, generator, save_path, perturb=False):
    
    # Getting x and y data from image generator
    X_test, y_test = generator.next()

    # Getting the class names associated with the ImageGenerator data.
    X_test_processed = [inc_net.preprocess_input(img) for img in X_test]
    
    # Our model's final layer has 1 output using the sigmoid function, the outputted prediction is in the form [[prediction]].
    # Lime requires a prediction in the form [[prediction, prediction]] regardless of if it is binary classification or not.
    # Lime given a prediction [prediction, prediction] will take the index of the highest value, the function then uses that index as the associated predicted class.
    # The predict_fn therefore has to ensure that if the prediction is > 0.5 the highest value should be in position 1 and of < 0.5 should be in position 0.
    def predict_fn(x):
        pred_list = model.predict(x)
        pred_list_final = []
        for index in range(len(pred_list)):
            prediction = pred_list[index][0]
            pred_list_final.append(np.insert(pred_list[index], 0, (1-prediction)))
        pred_list_final = np.array(pred_list_final)
        return pred_list_final

    # Initialising the LimeImageExplainer
    explainer = lime_image.LimeImageExplainer(verbose=False)

    # Generating random index which are used to randomly sample the X_test data and generate explanations.
    random_indexes = random.sample(range(1,len(X_test)),10) # <--- Change this value to alter the amount of generated explanations.

    for index in random_indexes:
        # Calling the explain_instance function, explaining the instance at the current index. 
        explanation = explainer.explain_instance(X_test[index].astype(np.float), predict_fn, top_labels = 2, hide_color = 0, num_samples = 1000)
        
        # Extracting the class names from the dataset.
        classes = list(generator.class_indices)

        # Calculating the prediction for the current instance.
        preds = predict_fn(np.expand_dims(X_test[index], axis=0))
        
        # Getting both labels in order to show positive and negative regions for both.
        class_pred_1 = int(np.where(np.argmax(preds[0])==1, 1,0))
        class_pred_2 = int(np.where(np.argmax(preds[0])==1, 0,1))

        # Presenting the explanation.
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle('Classifier result: {}'.format(classes[class_pred_1]))
                
        # Setting values to rows and column variables
        if perturb == False:
            rows = 2
        else:
            rows = 3
        
        columns = 3

        # Adding the subplot of the original image
        fig.add_subplot(rows, columns, 1)
        plt.imshow(X_test_processed[index])
        plt.rc('axes', titlesize=8) 
        plt.axis('off')
        plt.title('Original Image')

        # Calculating masks for the first label with only positive regions
        temp, mask = explanation.get_image_and_mask(class_pred_1, positive_only=True , num_features=5, hide_rest=False)
        # Adding the subplot for only positive regions for the first label
        fig.add_subplot(rows, columns, 2)
        plt.imshow(mark_boundaries(temp, mask))
        plt.rc('axes', titlesize=8)
        plt.axis('off')
        plt.title('Positive Regions for {}'.format(classes[class_pred_1]))
        
        # Calculating masks for the first label with both positive and negative regions
        temp, mask = explanation.get_image_and_mask(class_pred_1, positive_only=False, num_features=10, hide_rest=False)
        # Adding the subplot for only both positive and negative regions for the first label
        fig.add_subplot(rows, columns, 3)
        plt.imshow(mark_boundaries(temp, mask))
        plt.rc('axes', titlesize=8)
        plt.axis('off')
        plt.title('Positive & Negative Regions for {}'.format(classes[class_pred_1]))

        # Adding the subplot of the original image
        fig.add_subplot(rows, columns, 4)
        plt.imshow(X_test_processed[index])
        plt.rc('axes', titlesize=8)
        plt.axis('off')
        plt.title('Original Image')

        # Calculating masks for the second label with only positive regions
        temp, mask = explanation.get_image_and_mask(class_pred_2, positive_only=True , num_features=5, hide_rest=False)
        # Adding the subplot for only positive regions for the second label
        fig.add_subplot(rows, columns, 5)
        plt.imshow(mark_boundaries(temp, mask))
        plt.rc('axes', titlesize=8)
        plt.axis('off')
        plt.title('Positive Regions for {}'.format(classes[class_pred_2]))
        
        # Calculating masks for the second label with both positive and negative regions
        temp, mask = explanation.get_image_and_mask(class_pred_2, positive_only=False, num_features=10, hide_rest=False)
        # Adding the subplot for only both positive and negative regions for the second label
        fig.add_subplot(rows, columns, 6)
        plt.imshow(mark_boundaries(temp, mask))
        plt.rc('axes', titlesize=8)
        plt.axis('off')
        plt.title('Positive & Negative Regions for {}'.format(classes[class_pred_2]))

        if perturb == True:
            # Calling the transform function
            perturbed_img = image_perturbation(X_test_processed[index])
            # Adding a subplot for the perturbed image
            fig.add_subplot(rows, columns, 8)
            pred_peturb = round(model.predict(np.expand_dims(perturbed_img,axis=0))[0][0])
            plt.imshow(perturbed_img)
            plt.rc('axes', titlesize=8)
            plt.axis('off')
            plt.title("Peturbed classification: " + str(classes[pred_peturb]))

        # Saving explanation to the specified path
        plt.savefig(save_path+'explanation_'+str(index)+'.png')

############################################################
# INITIALISING MODEL & DATA FOR FAKE VS REAL FACES DATASET #
############################################################
train_generator_1, test_generator_1, valid_generator_1 = get_dataset_1() # Getting the generators associated with dataset 1
model_name = "xai_image_classification_data_1_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/lime_image_explanations/image_data_1/' # Declaring were to save the explanations
extracting_lime_explanation(model, test_generator_1, save_path, perturb=True) # Calling the explanation function

####################################################
# INITIALISING MODEL & DATA FOR CAT VS DOG DATASET #
####################################################
train_generator_2, test_generator_2 = get_dataset_2() # Getting the generators associated with dataset 2
model_name = "xai_image_classification_data_2_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/lime_image_explanations/image_data_2/' # Declaring were to save the explanations
extracting_lime_explanation(model, test_generator_2, save_path, perturb=True) # Calling the explanation function

###########################################################
# INITIALISING MODEL & DATA FOR WATERMARK VS NO_WATERMARK #
###########################################################
train_generator_3, test_generator_3 = get_dataset_3() # Getting the generators associated with dataset 2
model_name = "xai_image_classification_data_3_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/lime_image_explanations/image_data_3/' # Declaring were to save the explanations
extracting_lime_explanation(model, test_generator_3, save_path, perturb=True) # Calling the explanation function
