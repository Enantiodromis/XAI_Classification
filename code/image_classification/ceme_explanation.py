import random
import numpy as np
import tensorflow as tf
from aix360.algorithms.contrastive import CEMExplainer, KerasClassifier
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.keras.models import load_model
from image_classification_core import binary_dataset_creation, img_classification_model, plot_accuracy_loss

# Used to confirm if GPU is being leveraged for the running of the model.
# Uncomment line 12 and 13 to see which devices are being used by tensorflow.
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#######################################
# EXTRACTING CEMExplainer EXPLANATION #
#######################################
# +------------------------------------------------------+
# + Function inputs:                                     +
# +    - TensorFlow classification model                 +
# +    - A image_generator to create explanations from   +
# +------------------------------------------------------+
# + Function outputs:                                    +
# +     -                                                +
# +------------------------------------------------------+
def extracting_CEMExplainer_explanation(classifier_model, image_generator):
    # Initialising the CEMExplainer with our classification model
    explainer = CEMExplainer(classifier_model)

    # Generating a random list of indexes based on the length of the image_generator
    random_indexes = random.sample(range(1,len(image_generator)),10)
    #for index in random_indexes:
    #    plt.imshow(image_generator[index])

############################################
# INITIALISING MODEL & EXPLAINER VARIBALES #
############################################
# Creating the necessary image generators to train and test the model
train_generator, test_generator = binary_dataset_creation(32, 256, 256, False, True, file_path="datasets/image_data_2") 
model_name = "ceme_xai_image_classification_ConvNet" # Initialising a model name
#history, model = img_classification_model(train_generator, test_generator, 10, model_name) # Training the model

cat_dog_classification_model = load_model("models/ceme_xai_image_classification_ConvNet.h5") # Loading the saved model
history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
plot_accuracy_loss(history, 10, model_name) # Plotting the training history and saving the plots
