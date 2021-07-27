from operator import add
from pickle import FALSE
import random
from keras.engine.sequential import Sequential

import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
from aix360.algorithms.contrastive import CEMExplainer, KerasClassifier
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, MaxPool2D, Flatten, Dense
from keras.models import Model, Sequential
from keras.layers.convolutional import UpSampling2D
from keras.models import load_model
from keras.applications import inception_v3 as inc_net
from tensorflow.python.ops.gen_math_ops import Max
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from PIL import Image, ImageOps


from image_classification_core import (binary_dataset_creation,
                                       img_classification_model,
                                       plot_accuracy_loss)

# Used to confirm if GPU is being leveraged for the running of the model.
# Uncomment line 16 and 17 to see which devices are being used by tensorflow.
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
def extracting_CEMExplainer_explanation(classifier_model, image_generator, autoencoder):
    number_of_images = 10
    # Wrap model into a framework independent class structure
    model = KerasClassifier(classifier_model)

    # Initialising the CEMExplainer with our classification model
    explainer = CEMExplainer(model)

    # Extracting the image path lists from the image 
    path_list = []
    for file in image_generator.filenames:
        path_list.append("datasets/image_data_2/"+file)

    images = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(256, 256))
        x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = inc_net.preprocess_input(x)
        images.append(x)
    input_image = images[1]

    print(np.min(input_image))
    print(np.max(input_image))
    plt.imshow((input_image[:,:,0] + 0.5)*255, cmap="gray")
    plt.show()
    print("Predicted class:", model.predict_classes(np.expand_dims(input_image, axis=0)))
    print("Predicted logits:", model.predict(np.expand_dims(input_image, axis=0)))

    # Obtaining pertinent negative (PN) explanation
    arg_mode = "PN" # Find pertinent negative
    arg_max_iter = 1000 # Maximum number of iterations to search for the optimal PN for given parameter settings
    arg_init_const = 10.0 # Initial coefficient value for main loss term that encourages class change
    arg_b = 9 # Number of updates to the coefficient of the main loss term
    arg_kappa = 0.5 # Minimum confidence gap between the PNs (changed) class probability and original class probability
    arg_beta = 1e-1 # Controls sparsity of the solution (L1 loss)
    arg_gamma = 100 # Controls how much to adhere to a (optionally trained) autoencoder
    arg_alpha = 0.01 # Penalizes L2 norm of the solution
    arg_threshold = 0.05 # Automatically turn off features <= arg_threshold if arg_threshold < 1
    arg_offset = 0.5 # The model assumes clssifier trained on data normalised 
                                                                                                                  
    (adv_pn, delta_pn, info_pn) = explainer.explain_instance(np.expand_dims(images[1], axis=0), arg_mode, autoencoder, arg_kappa, arg_b, arg_max_iter, arg_init_const, arg_beta, arg_gamma)

    print(info_pn)

    arg_mode = "PP"  # Find pertinent positive
    arg_beta = 0.1 # Controls sparsity of the solution (L1 loss)

    (adv_pp, delta_pp, info_pp) = explainer.explain_instance(np.expand_dims(input_image, axis=0), arg_mode, autoencoder, arg_kappa, arg_b, arg_max_iter, arg_init_const, arg_beta, arg_gamma, arg_alpha, arg_threshold, arg_offset)

    print(info_pp)

    # rescale values from [-0.5, 0.5] to [0, 255] for plotting
    fig0 = (input_image[:,:,0] + 0.5)*255

    fig1 = (adv_pn[0,:,:,0] + 0.5) * 255
    fig2 = (fig1 - fig0) #rescaled delta_pn
    fig3 = (adv_pp[0,:,:,0] + 0.5) * 255
    fig4 = (delta_pp[0,:,:,0] + 0.5) * 255 #rescaled delta_pp

    f, axarr = plt.subplots(1, 5, figsize=(10,10))
    axarr[0].set_title("Original" + "(" + str(model.predict_classes(np.expand_dims(input_image, axis=0))[0]) + ")")
    axarr[1].set_title("Original + PN" + "(" + str(model.predict_classes(adv_pn)[0]) + ")")
    axarr[2].set_title("PN")
    axarr[3].set_title("Original + PP")
    axarr[4].set_title("PP" + "(" + str(model.predict_classes(delta_pp)[0]) + ")")

    axarr[0].imshow(fig0, cmap="gray")
    axarr[1].imshow(fig1, cmap="gray")
    axarr[2].imshow(fig2, cmap="gray")
    axarr[3].imshow(fig3, cmap="gray")
    axarr[4].imshow(fig4, cmap="gray")
    plt.show()

def conv_autoencoder(train_generator, test_generator, epochs, model_name):
    input = Input(shape=(256, 256, 3))

    # Encoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return autoencoder

def normalising_img_data(path_list, saving_file):
    images = []
    for img_path in path_list[:2]:
        img = Image.open(img_path)
        img = ImageOps.grayscale(img)
        img = np.array(img)
        img = img.astype('float32')
        img = (img / 255.0)-0.5
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save("test.jpg")
        images.append(img)
    #print(preprocessing.minmax_scale(input_image, feature_range=(-0.5, 0.5)))


############################################
# INITIALISING MODEL & EXPLAINER VARIBALES #
############################################
# Creating the necessary image generators to train and test the model
train_generator, test_generator = binary_dataset_creation(32, 256, 256, False, True, file_path="datasets/image_data_2") 
model_name = "ceme_xai_image_classification_ConvNet" # Initialising a model name
#history, model = img_classification_model(train_generator, test_generator, 10, model_name) # Training the model

cat_dog_classification_model = load_model("models/ceme_xai_image_classification_ConvNet.h5") # Loading the saved model
autoencoder = conv_autoencoder(train_generator, test_generator, 10, model_name)

path_list = []
for file in train_generator.filenames:
    path_list.append("datasets/image_data_2/"+file)

normalising_img_data(path_list, "datasets/image_data_2/" )

"""import os
from PIL import Image
folder_path = 'datasets/image_data_2'
extensions = []
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        print('** Path: {}  **'.format(file_path), end="\r", flush=True)
        im = Image.open(file_path)
        rgb_im = im.convert('RGB')
        if filee.split('.')[1] not in extensions:
            extensions.append(filee.split('.')[1])"""
#extracting_CEMExplainer_explanation(cat_dog_classification_model, train_generator, autoencoder)

#labels_2 = list(valid_generator.class_indices.keys())

# history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
# plot_accuracy_loss(history, 10, model_name) # Plotting the training history and saving the plots
