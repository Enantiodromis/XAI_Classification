import random
from re import X

import numpy as np
import pandas as pd
from keras.applications import inception_v3 as inc_net
from keras.models import load_model

from lime import lime_image
from matplotlib import pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import array_to_img as ati
from keras.applications.imagenet_utils import decode_predictions

from image_classification_core import (binary_dataset_creation,
                                       img_classification_model,
                                       plot_accuracy_loss)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Used to confirm if GPU is being leveraged for the running of the model.
# Uncomment line 12 and 13 to see which devices are being used by tensorflow.
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

###############################
# EXTRACTING LIME EXPLANATION #
###############################
# +------------------------------------------------------+
# + Function inputs:                                     +
# +    - TensorFlow classification model                 +
# +    - A image_generator to create explanations from   +
# +------------------------------------------------------+
# + Function outputs:                                    +
# +     -                                                +
# +------------------------------------------------------+
def extracting_lime_explanation(model, generator, X_test, y_test):

    def lime_explainer_image():
        # Message 
        print('This may take a few minutes...')

        X_test_processed = [inc_net.preprocess_input(img) for img in X_test]
        y_test_processed = [label.astype(np.uint8) for label in y_test]

        def predict_fn(x):
            return model.predict_proba(x)
        
        # Create explainer 
        explainer = lime_image.LimeImageExplainer(verbose=False)
        from skimage.segmentation import mark_boundaries  
        random_indexes = random.sample(range(1,len(X_test)),3)

        for index in random_indexes:
            # Set up the explainer
            explanation = explainer.explain_instance(X_test[index].astype(np.float), predict_fn, top_labels = 2, hide_color = 0, num_samples = 10000)
            ati(X_test[index])

            labels = list(generator.class_indices)
            preds = model.predict(np.expand_dims(X_test_processed[index],axis=0))
            class_pred = model.predict_classes(np.expand_dims(X_test_processed[index],axis=0))[0][0]
            pct = np.max(preds, axis=-1)[0]

            print("LABELS ", labels)
            print("CLASS PRED ", class_pred)
            print("PREDICTION", preds)
            print(y_test_processed[index])
            print("TOP LABELS: ", explanation.top_labels)
            print("LOCAL PRED: ", explanation.local_pred)
            class_pred_2 = np.where(preds[0][0] > 0.5, 1,0)

            temp, mask = explanation.get_image_and_mask(0, positive_only=True , num_features=5, hide_rest=False)
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (10, 4))

            fig.suptitle('Classifier result: {}'.format(labels[class_pred_2]))
            fig.tight_layout(h_pad=2)

            ax1.imshow(X_test_processed[index])
            ax1.set_title('Original Image')
            ax1.axis('off')

            ax2.imshow(mark_boundaries(temp, mask))
            ax2.set_title('Positive Regions for {}'.format(labels[class_pred_2]))
            ax1.axis('off')

            temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=10, hide_rest=False)
            ax3.imshow(mark_boundaries(temp, mask))
            ax3.set_title('Positive & Negative Regions for {}'.format(labels[class_pred_2]))
            ax1.axis('off')

            fig.savefig("image_explanations/lime_image_explanations/explanation_"+str(labels[class_pred_2])+"_"+str(index)+".jpg")
    lime_explainer_image()

############################################################
# INITIALISING MODEL & DATA FOR FAKE VS REAL FACES DATASET #
############################################################
folder_path_train = 'datasets/image_data/image_data_1/real_vs_fake/train'
folder_path_test = 'datasets/image_data/image_data_1/real_vs_fake/test'
folder_path_valid = 'datasets/image_data/image_data_1/real_vs_fake/valid'

train_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_train)
test_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_test)
valid_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_valid)

X_test, y_test = next(valid_generator)

model_name = "lime_xai_image_classification_data_1_ConvNet"
#history, model = img_classification_model(train_generator, valid_generator, 60, model_name)
model = load_model("models/image_models/lime_xai_image_classification_data_1_ConvNet.h5")

extracting_lime_explanation(model, valid_generator, X_test, y_test)
#history=np.load('model_history/image_classification_ConvNet_image_data_1.npy',allow_pickle='TRUE').item()
#plot_accuracy_loss(history, 50)

####################################################
# INITIALISING MODEL & DATA FOR CAT VS DOG DATASET #
####################################################
folder_path = 'datasets/image_data/image_data_2'
train_generator, test_generator = binary_dataset_creation(32, 256, 256, False, True, file_path=folder_path)

X_test, y_test = next(test_generator)

model_name = "lime_xai_image_classification_data_2_ConvNet" # Initialising a model name
#history, model = img_classification_model(train_generator, test_generator, 60, model_name) # Training the model
history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model

extracting_lime_explanation(model, test_generator, X_test, y_test)
#history=np.load('model_history/image_classification_ConvNet_image_data_1.npy',allow_pickle='TRUE').item()
#plot_accuracy_loss(history, 50)

###########################################################
# INITIALISING MODEL & DATA FOR WATERMARK VS NO_WATERMARK #
###########################################################
folder_path_train = 'datasets/image_data/image_data_3/train'
folder_path_valid = 'datasets/image_data/image_data_3/valid'


train_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_train)
valid_generator = binary_dataset_creation(32, 256, 256, False, False, file_path=folder_path_valid)

X_test, y_test = next(valid_generator)

model_name = "lime_xai_image_classification_data_3_ConvNet"
#history, model = img_classification_model(train_generator, valid_generator, 60, model_name)
model = load_model("models/image_models/lime_xai_image_classification_data_1_ConvNet.h5")

extracting_lime_explanation(model, valid_generator, X_test, y_test)
#history=np.load('model_history/image_classification_ConvNet_image_data_1.npy',allow_pickle='TRUE').item()
#plot_accuracy_loss(history, 50)
