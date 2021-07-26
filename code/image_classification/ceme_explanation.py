import os
import random
import sys
import warnings

import numpy as np
import tensorflow as tf
from aix360.algorithms.contrastive import CEMExplainer, KerasClassifier
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.keras.models import load_model
from image_classification_core import binary_dataset_creation, img_classification_model, plot_accuracy_loss

warnings.filterwarnings('ignore')

#######################################
# EXTRACTING CEMExplainer EXPLANATION #
#######################################
def extracting_CEMExplainer_explanation(classifier_model, image_generator):
    # Initialising explainer object
    explainer = CEMExplainer(classifier_model)

    # Generating a random list of indexes based on the length of the image_generator
    random_indexes = random.sample(range(1,len(image_generator)),10)
    #for index in random_indexes:
    #    plt.imshow(image_generator[index])

train_generator, test_generator = binary_dataset_creation(32, 256, 256, False, True, file_path="datasets/image_data_2")
model_name = "ceme_xai_image_classification_ConvNet"
history, model = img_classification_model(train_generator, test_generator, 10, model_name)

model_2 = load_model("models/ceme_xai_image_classification_ConvNet.h5")
history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item()
plot_accuracy_loss(history, 10, model_name)
