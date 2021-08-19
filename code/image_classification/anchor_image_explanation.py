from tensorflow.keras.backend import expand_dims
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import inception_v3 as inc_net
from alibi.explainers import AnchorImage

from image_dataset_1_processing import get_dataset_1
from image_dataset_2_processing import get_dataset_2
from image_dataset_3_processing import get_dataset_3

import alibi

from skimage import transform

def anchors_image_explainer(model, train_generator, test_generator):

    X_train, y_train = train_generator.next()
    X_test, y_test = test_generator.next()

    classes = list(train_generator.class_indices)
    X_train_processed = [inc_net.preprocess_input(img) for img in X_train]
    X_test_processed = [inc_net.preprocess_input(img) for img in X_test]
    
    def wrapped_predict(x):
        pred_list = model.predict(np.expand_dims(x, axis=0))
        pred = float(pred_list[0][0])
        if pred > 0.5:
            prediction = np.insert(pred_list[0], 0, (1-pred))
        else: 
            prediction = np.insert(pred_list[0], 1, (pred*1))
        return prediction
    
    def wrapped_predict_all(payloads):
        results = [wrapped_predict(payload) for payload in payloads]
        prediction = np.array(results, dtype=float)
        print(prediction)
        return prediction

    predict_fn = lambda x: wrapped_predict_all(x)

    random_indexes = random.sample(range(1,len(X_test)),2)

    for index in random_indexes:
        image_shape = X_train_processed[1].shape
        kwargs = {'n_segments':15, 'compactness':20, 'sigma':.5}
        explainer = AnchorImage(predict_fn, image_shape, segmentation_fn='slic',
                                segmentation_kwargs=kwargs, images_background=None)
        image = X_test_processed[index]
        explanation = explainer.explain(image, threshold=0.1, p_sample=0.4, tau=0.55)

        # create figure
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle("Classification: " + str(classes[int(y_test[index])]))
                
        # setting values to rows and column variables
        rows = 1
        columns = 3

        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 1)
            
        # showing image
        plt.imshow(X_test_processed[index])
        plt.rc('axes', titlesize=8) 
        plt.axis('off')
        plt.title('Original Image')
            
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
            
        # showing image
        plt.imshow(explanation.segments)
        plt.rc('axes', titlesize=8)
        plt.axis('off')
        plt.title('Anchor')

        transformed_img = transform.rotate(X_test_processed[index], angle=-50, cval=255)
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 3)
            
        # showing image
        pred_peturb = model.predict(np.expand_dims(transformed_img,axis=0))
        plt.imshow(transformed_img)
        plt.rc('axes', titlesize=8)
        plt.axis('off')
        plt.title("Peturbed classification: " + str(pred_peturb))

        plt.savefig("image_explanations/anchors_image_explanations/image_data_1/forward/anchors_explainer_"+str(index)+".png")

############################################################
# INITIALISING MODEL & DATA FOR FAKE VS REAL FACES DATASET #
############################################################
train_generator, test_generator, valid_generator = get_dataset_1()
model_name = "xai_image_classification_data_1_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/anchor_image_explanations/image_data_1/'
anchors_image_explainer(model, train_generator, valid_generator)

####################################################
# INITIALISING MODEL & DATA FOR CAT VS DOG DATASET #
####################################################
train_generator, test_generator = get_dataset_2()
model_name = "xai_image_classification_data_2_ConvNet" # Initialising a model name
model = load_model("models/image_models/"+model_name+".h5") # Loading the saved model
save_path = 'image_explanations/anchor_image_explanations/image_data_2/'
anchors_image_explainer(model, train_generator, test_generator)

###########################################################
# INITIALISING MODEL & DATA FOR WATERMARK VS NO_WATERMARK #
###########################################################
train_generator, valid_generator = get_dataset_3()
model_name = "xai_image_classification_data_3_ConvNet"
model = load_model("models/image_models/"+model_name+".h5")
save_path = 'image_explanations/anchor_image_explanations/image_data_3/'
anchors_image_explainer(model, train_generator, valid_generator)


