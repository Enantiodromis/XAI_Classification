import random
from re import X

import numpy as np
import pandas as pd
from keras.applications import inception_v3 as inc_net
from keras.models import load_model

from lime import lime_image
from matplotlib import pyplot as plt

from dataset_processing.image_dataset_1_processing import get_dataset_1
from dataset_processing.image_dataset_2_processing import get_dataset_2
from dataset_processing.image_dataset_3_processing import get_dataset_3


###############################
# EXTRACTING LIME EXPLANATION #
###############################
def extracting_lime_explanation(model, generator, save_path):
        # Message 
        print('This may take a few minutes...')
        X_test, y_test = generator.next()

        X_test_processed = [inc_net.preprocess_input(img) for img in X_test]
        y_test_processed = [label.astype(np.uint8) for label in y_test]

        def predict_fn(x):
            pred_list = model.predict(x)
            pred_list_final = []
            for index in range(len(pred_list)):
                prediction = pred_list[index][0]
                if prediction > 0.5:
                    pred_list_final.append(np.insert(pred_list[index], 0, (1-prediction)))
                else: 
                    pred_list_final.append(np.insert(pred_list[index], 1, (1-prediction)))
            pred_list_final = np.array(pred_list_final)
            return pred_list_final
        
        # Create explainer 
        explainer = lime_image.LimeImageExplainer(verbose=False)
        from skimage.segmentation import mark_boundaries  
        random_indexes = random.sample(range(1,len(X_test)),10)

        for index in random_indexes:
            # Set up the explainer
            explanation = explainer.explain_instance(X_test[index].astype(np.float), predict_fn, top_labels = 2, hide_color = 0, num_samples = 10000)

            labels = list(generator.class_indices)
            preds = model.predict(np.expand_dims(X_test[index], axis=0))
            class_pred_2 = int(np.where(preds[0][0] > 0.5, 1,0))

            temp, mask = explanation.get_image_and_mask(class_pred_2, positive_only=True , num_features=5, hide_rest=False)
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (10, 4))

            fig.suptitle('Classifier result: {}'.format(labels[class_pred_2]))
            fig.tight_layout(h_pad=2)

            ax1.imshow(X_test_processed[index])
            ax1.set_title('Original Image')
            ax1.axis('off')

            ax2.imshow(mark_boundaries(temp, mask))
            ax2.set_title('Positive Regions for {}'.format(labels[class_pred_2]))
            ax1.axis('off')

            temp, mask = explanation.get_image_and_mask(class_pred_2, positive_only=False, num_features=10, hide_rest=False)
            ax3.imshow(mark_boundaries(temp, mask))
            ax3.set_title('Positive & Negative Regions for {}'.format(labels[class_pred_2]))
            ax1.axis('off')

            fig.savefig(save_path+str(labels[class_pred_2])+"_"+str(index)+".jpg")

############################################################
# INITIALISING MODEL & DATA FOR FAKE VS REAL FACES DATASET #
############################################################
train_generator, test_generator, valid_generator = get_dataset_1()
model_name = "lime_xai_image_classification_data_1_ConvNet"
model = load_model("models/image_models/lime_xai_image_classification_data_1_ConvNet.h5")
save_path = 'image_explanations/lime_image_explanations/image_data_1/'
extracting_lime_explanation(model, valid_generator, save_path)

####################################################
# INITIALISING MODEL & DATA FOR CAT VS DOG DATASET #
####################################################
train_generator, test_generator = get_dataset_2()
model_name = "lime_xai_image_classification_data_2_ConvNet" 
model = load_model("models/image_models/"+model_name+".h5")
save_path = 'image_explanations/lime_image_explanations/image_data_2/'
extracting_lime_explanation(model, test_generator, save_path)

###########################################################
# INITIALISING MODEL & DATA FOR WATERMARK VS NO_WATERMARK #
###########################################################
train_generator, valid_generator = get_dataset_3()
model_name = "lime_xai_image_classification_data_3_ConvNet"
model = load_model("models/image_models/lime_xai_image_classification_data_1_ConvNet.h5")
save_path = 'image_explanations/lime_image_explanations/image_data_3/'
extracting_lime_explanation(model, valid_generator, save_path)
