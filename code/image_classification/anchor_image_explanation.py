# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: anchor_image_explanation
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
import random

import matplotlib.pyplot as plt
import numpy as np
from alibi.explainers import AnchorImage
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.models import load_model

from image_dataset_1_processing import get_dataset_1
from image_dataset_2_processing import get_dataset_2
from image_dataset_3_processing import get_dataset_3

from image_perturbation import image_perturbation

#####################################
# ANCHOR IMAGE EXPLANATION FUNCTION #
#####################################
def extracting_anchors_explanation(model, train_generator, test_generator, perturb = False):
    
    # Getting x and y data from image generator
    X_train, y_train = train_generator.next()
    X_test, y_test = test_generator.next()

    # Getting the class names associated with the ImageGenerator data.
    classes = list(train_generator.class_indices)

    # Processing the image data of the x data extracted from the imagegenerator in lines 19-20 using the inception_v3 model.
    X_train_processed = [inc_net.preprocess_input(img) for img in X_train]
    X_test_processed = [inc_net.preprocess_input(img) for img in X_test]
    
    # Our model's final layer has 1 output using the sigmoid function, the outputted prediction is in the form [[prediction]].
    # Anchors requires a prediction in the form [[prediction, prediction]] regardless of if it is binary classification or not.
    # Anchors given a prediction [prediction, prediction] will take the index of the highest value, the function then uses that index as the associated predicted class.
    # The wrapper therefore has to ensure that if the prediction is > 0.5 the highest value should be in position 1 and of < 0.5 should be in position 0.
    def wrapped_predict(x):
        pred_list = model.predict(np.expand_dims(x, axis=0)) # Calculating the prediction for the inputted instance.
        pred = float(pred_list[0][0]) # Getting the value of the prediction from the returned [[prediction]] format.
        prediction = np.insert(pred_list[0], 0, (1-pred)) # Ensuring that the largest value is in the correct index for the given prediction.
        return prediction
    
    # Anchors sends "payloads" of instances to be predicted, this function, sends individual instances from a payload to be predicted by the wrapped_predict function.
    def wrapped_predict_all(payloads):
        results = [wrapped_predict(payload) for payload in payloads]
        prediction = np.array(results, dtype=float)
        return prediction

    predict_fn = lambda x: wrapped_predict_all(x)

    # Generating random index which are used to randomly sample the X_test data and generate explanations.
    random_indexes = random.sample(range(1,len(X_test)),10) # <--- Change this value to alter the amount of generated explanations.

    for index in random_indexes:
        image_shape = X_train_processed[1].shape
        kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5} # Arguments for the segmentation algorithm

        # Initialising the AnchorImage explainer, using our defined predition wrapper, the image shape,
        # the selected segementation algorithm and the segmentation algorithms arguments
        explainer = AnchorImage(predict_fn, image_shape, segmentation_fn='slic',
                                segmentation_kwargs=kwargs, images_background=None)

        image = X_test_processed[index] # Extracting the image which we will use to explain.

        # Explaining the classification of the instance and defining the explainer parameters.
        explanation = explainer.explain(image, threshold=0.5, p_sample=0.1, tau=0.05) 

        # Presenting the explanation.
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle("Classification: " + str(classes[int(y_test[index])]))
        
        # Setting values for rows and column
        rows = 1
        if perturb == False:
            columns = 2
        else:
            columns = 3

        # Adding the subplot of the original image
        fig.add_subplot(rows, columns, 1)
        plt.imshow(X_test_processed[index])
        plt.rc('axes', titlesize=8) 
        plt.axis('off')
        plt.title('Original Image')
            
        # Adding a subplot of the explanation
        fig.add_subplot(rows, columns, 2)
        plt.imshow(explanation.segments)
        plt.rc('axes', titlesize=8)
        plt.axis('off')
        plt.title('Anchor')

        if perturb == True:
            peturbed_img = image_perturbation(X_test_processed[index])
            # Adding a subplot of the perturbed image
            fig.add_subplot(rows, columns, 3)
            pred_peturb = model.predict(np.expand_dims(peturbed_img,axis=0))[0][0]
            plt.imshow(peturbed_img)
            plt.rc('axes', titlesize=8)
            plt.axis('off')
            plt.title("Peturbed classification: " + str(pred_peturb))
        
        # Saving explanation to the specified path
        plt.savefig(save_path+'explanation_'+str(index)+'.png')

############################################################
# INITIALISING MODEL & DATA FOR FAKE VS REAL FACES DATASET #
############################################################
train_generator_1, test_generator_1, valid_generator_1 = get_dataset_1() # Getting the generators associated with dataset 1
model_name = "xai_image_classification_data_1_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/anchors_image_explanations/image_data_1/' # Declaring were to save the explanations
extracting_anchors_explanation(model, train_generator_1, test_generator_1, perturb=True) # Calling the explanation function

####################################################
# INITIALISING MODEL & DATA FOR CAT VS DOG DATASET #
####################################################
train_generator_2, test_generator_2 = get_dataset_2() # Getting the generators associated with dataset 2
model_name = "xai_image_classification_data_2_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/anchors_image_explanations/image_data_2/' # Declaring were to save the explanations
extracting_anchors_explanation(model, train_generator_2, test_generator_2, perturb=True) # Calling the explanation function

###########################################################
# INITIALISING MODEL & DATA FOR WATERMARK VS NO_WATERMARK #
###########################################################
train_generator_3, test_generator_3 = get_dataset_3() # Getting the generators associated with dataset 3
model_name = "xai_image_classification_data_3_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/anchors_image_explanations/image_data_3/' # Declaring were to save the explanations
extracting_anchors_explanation(model, train_generator_3, test_generator_3, perturb=True) # Calling the explanation function


