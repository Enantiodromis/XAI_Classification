###########
# IMPORTS #
###########
from time import time
from keras.backend import exp
import numpy as np 
import os 
import PIL
import PIL.Image
from numpy.core.fromnumeric import ravel
from scipy.ndimage import interpolation
from skimage import segmentation
from skimage.color.colorconv import gray2rgb, rgb2gray
from sklearn.utils import shuffle
import tensorflow as tf
import pathlib
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, MaxPool2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model
import pickle
from lime import lime_image
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from skimage.color import label2rgb
from lime.wrappers.scikit_image import SegmentationAlgorithm
import random

################
# DATA DETAILS #
################
def data_details(data_path):
    data_dir = pathlib.Path(data_path)
    tt_list = ["train", "test", "valid"]
    for test_or_train in tt_list:
        if test_or_train == "train":
            print("#########################")
            print("# Training data details #")
            print("#########################")
        if test_or_train == "test":
            print("########################")
            print("# Testing data details #")
            print("########################")
        if test_or_train == "valid":
            print("###########################")
            print("# Validation data details #")
            print("###########################")

        real_path = test_or_train+'/real/*.jpg'
        image_count_real = len(list(data_dir.glob(real_path)))
        print("Number of real images in the %s dataset:" % test_or_train, image_count_real)
        
        fake_path = test_or_train+'/fake/*.jpg'
        image_count_fake = len(list(data_dir.glob(fake_path)))
        print("Number of fake images in the %s dataset:" % test_or_train, image_count_fake)
        print("Total number of images:", image_count_real+image_count_fake)
        print("")

####################
# DATASET CREATION #
####################
def binary_dataset_creation(batch_size, img_height, img_width, from_dataframe, need_train_test_split, dataframe = None, file_path = None):
    if need_train_test_split == False:
        data_generator = ImageDataGenerator()
        if from_dataframe == True:
            generator = data_generator.flow_from_dataframe(
                dataframe = dataframe,
                x_col= 'path',
                y_col= 'label_str',
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                shuffle=True
            )
        else:
            generator = data_generator.flow_from_directory(
                directory = file_path,
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                shuffle=True   
            )
        return generator
    else: 
        data_generator = ImageDataGenerator(validation_split=0.2)
        if from_dataframe == True:
            generator_1 = data_generator.flow_from_dataframe(
                dataframe = dataframe,
                x_col= 'path',
                y_col= 'label_str',
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="training",
                shuffle=True
            )
            generator_2 = data_generator.flow_from_dataframe(
                dataframe = dataframe,
                x_col= 'path',
                y_col= 'label_str',
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="validation",
                shuffle=True
            )
        else:
            generator_1 = data_generator.flow_from_directory(
                directory = file_path,
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="training",
                shuffle=True   
            )
            generator_2 = data_generator.flow_from_directory(
                directory = file_path,
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="validation",
                shuffle=True   
            )
        return generator_1, generator_2

#######################################
# BUILD, TRAIN AND EVALUATE THE MODEL #
#######################################
def img_classification_model(train_generator, test_generator, number_epochs, files_name):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=1e-4),
                metrics=['acc'])

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=number_epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
    )
    np.save('model_history/'+files_name+'.npy',history.history)
    model.save('models/'+files_name+'.h5') 

    return history, model

###############################
# EXTRACTING LIME EXPLANATION #
###############################
def extracting_lime_explanation(model, path_list, labels):

    def transform_img_fn(path_list):
        out = []
        for img_path in path_list:
            img = image.load_img(img_path, target_size=(256, 256))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = inc_net.preprocess_input(x)
            out.append(x)
        return np.vstack(out)
    from keras.applications.imagenet_utils import decode_predictions
    images = transform_img_fn(path_list)

    def lime_explainer_image():
        # Message
        print('This may take a few minutes...')
        
        # Create explainer 
        explainer = lime_image.LimeImageExplainer()
        #segmenter = SegmentationAlgorithm('slic', n_segments = 1000, compactness = 1, sigma = 1)

        from skimage.segmentation import mark_boundaries  
        
        random_indexes = random.sample(range(1,len(images)),10)
        for index in random_indexes:

            # Set up the explainer
            explanation = explainer.explain_instance(images[index].astype('double'), classifier_fn= model.predict,labels=(0,1),
                                                top_labels = 2, hide_color = 0, num_samples = 1000)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (10, 4))

            preds = explanation.local_pred
            prediction = np.argmax(preds)
            pct = np.max(preds)

            fig.suptitle('Classifier result: %r %% certainty of %r' %(round(pct,2)*100,labels[prediction]))
            fig.tight_layout(h_pad=2)

            ax1.imshow(images[index])
            ax1.set_title('Original Image')

            ax2.imshow(mark_boundaries(temp, mask))
            #ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
            ax2.set_title('Positive Regions for {}'.format(labels[explanation.top_labels[0]]))

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
            ax3.imshow(mark_boundaries(temp, mask))
            #ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
            ax3.set_title('Positive & Negative Regions for {}'.format(labels[explanation.top_labels[0]]))

            fig.savefig("image_explanations/lime_image_explanations/explanation_"+str(labels[explanation.top_labels[0]])+"_"+str(index)+".jpg")
    lime_explainer_image()

def plot_accuracy_loss(model_history, number_epochs, model_name):
    x_list = []
    x_list.extend(range(number_epochs))

    plt.figure(2,figsize=(15,4))
    plt.plot(model_history['acc'])
    plt.plot(model_history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks(x_list)
    plt.tight_layout()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("data_plots/"+model_name+"_acc.jpg")

    plt.figure(3,figsize=(15,4))
    plt.plot(model_history['loss'], color='green')
    plt.plot(model_history['val_loss'], color='red')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(x_list)
    plt.tight_layout()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("data_plots/"+model_name+"_loss.jpg")

test_df = pd.read_csv('datasets/image_data_1/test.csv')
train_df = pd.read_csv('datasets/image_data_1/train.csv')
valid_df = pd.read_csv('datasets/image_data_1/valid.csv')

test_generator = binary_dataset_creation(32, 256, 256, True, False, dataframe=test_df)
train_generator = binary_dataset_creation(32, 256, 256, True, False, dataframe=train_df)
valid_generator = binary_dataset_creation(32, 256, 256, True, False, dataframe=valid_df)

epochs = 10
#path_list = valid_df['path'].tolist()
#history, model = img_classification_model(train_generator, valid_generator, epochs, "image_classification_ConvNet_image_data_1")
#model = load_model("models/image_classification_ConvNet_image_data_1.h5")
#history=np.load('model_history/image_classification_ConvNet_image_data_1.npy',allow_pickle='TRUE').item()
#plot_accuracy_loss(history, epochs)
#extracting_lime_explanation()

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

train_generator, valid_generator = binary_dataset_creation(32, 256, 256, False, True, file_path="datasets/image_data_2")

model_name = "image_classification_ConvNet_image_data_2"
#history, model = img_classification_model(train_generator, valid_generator, epochs, model_name)
model = load_model("models/"+model_name+".h5")
history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item()
plot_accuracy_loss(history, epochs, model_name)

x_test, y_test = next(valid_generator)

path_list = []
for file in valid_generator.filenames:
    path_list.append("datasets\\image_data_2\\"+file)

labels = list(train_generator.class_indices.keys())
extracting_lime_explanation(model, path_list, labels)


