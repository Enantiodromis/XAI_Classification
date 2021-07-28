import random

import numpy as np
import pandas as pd
from keras.applications import inception_v3 as inc_net
from keras.models import load_model
from keras.preprocessing import image
from lime import lime_image
from matplotlib import pyplot as plt

from image_classification_core import (binary_dataset_creation,
                                       img_classification_model,
                                       plot_accuracy_loss)

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
def extracting_lime_explanation(model, path_list, labels):

    #def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    images = np.vstack(out)
    #images = transform_img_fn(path_list)

    def lime_explainer_image():
        # Message
        print('This may take a few minutes...')
        
        # Create explainer 
        explainer = lime_image.LimeImageExplainer()
        from skimage.segmentation import mark_boundaries  
        random_indexes = random.sample(range(1,len(images)),2)

        for index in random_indexes:
            # Set up the explainer
            explanation = explainer.explain_instance(images[index].astype('double'), classifier_fn = model.predict, labels = (0,1),top_labels = 2, hide_color = 0, num_samples = 1000)

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
            ax2.set_title('Positive Regions for {}'.format(labels[explanation.top_labels[0]]))

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
            ax3.imshow(mark_boundaries(temp, mask))
            ax3.set_title('Positive & Negative Regions for {}'.format(labels[explanation.top_labels[0]]))

            fig.savefig("image_explanations/lime_image_explanations/explanation_"+str(labels[explanation.top_labels[0]])+"_"+str(index)+".jpg")
    lime_explainer_image()

test_df = pd.read_csv('datasets/image_data_1/test.csv')
train_df = pd.read_csv('datasets/image_data_1/train.csv')
valid_df = pd.read_csv('datasets/image_data_1/valid.csv')

test_generator = binary_dataset_creation(32, 256, 256, True, False, dataframe=test_df)
train_generator = binary_dataset_creation(32, 256, 256, True, False, dataframe=train_df)
valid_generator = binary_dataset_creation(32, 256, 256, True, False, dataframe=valid_df)

path_list_1 = valid_df['path'].tolist()
labels_1 = list(valid_generator.class_indices.keys())
model_name = "lime_xai_image_classification_ConvNet"
model = load_model("models/lime_xai_image_classification_ConvNet.h5")

#extracting_lime_explanation(model, path_list_1, labels_1)
#history, model = img_classification_model(train_generator, valid_generator, 50, model_name)

extracting_lime_explanation(model, path_list_1, labels_1)
#history=np.load('model_history/image_classification_ConvNet_image_data_1.npy',allow_pickle='TRUE').item()
#plot_accuracy_loss(history, 50)
