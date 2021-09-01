# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: lime_text_explanation
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
import random

import keras.backend
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

from text_dataset_1_processing import get_dataset_1
from text_dataset_2_processing import get_dataset_2
from text_dataset_3_processing import get_dataset_3

from text_perturbation import text_perturbation

####################################
# LIME TEXT EXPLANATION FUNCTION #
####################################
def lime_text_explainer(X_test, model, word_index, tokenizer, max_len, class_names, save_path, perturb=False):
    
    # Creating a reverse word dictionary
    reverse_word_map = dict(map(reversed, word_index.items()))

    # Function takes a tokenized sentence and returns the words of the scentence.
    def sequence_to_text(list_of_indices):
        # Iterating over the tokenized sentence and returning the associated words in a list.
        words = [str(reverse_word_map.get(word)) for word in list_of_indices]
        return words
    my_texts = np.array(list(map(sequence_to_text, X_test))) # "Un-tokenizing" X_test using the function defined above.
    
    # Our model's final layer has 1 output using the sigmoid function, the outputted prediction is in the form [[prediction]].
    # Lime requires a prediction in the form [[prediction, prediction]] regardless of if it is binary classification or not.
    # Lime given a prediction [prediction, prediction] will take the index of the highest value, the function then uses that index as the associated predicted class.
    # The predict_fn therefore has to ensure that if the prediction is > 0.5 the highest value should be in position 1 and of < 0.5 should be in position 0.
    def wrapped_predict(strings):
        cnn_rep = tokenizer.texts_to_sequences(strings)
        text_data = pad_sequences(cnn_rep, maxlen=max_len)
        pred_list = model.predict(text_data)
        pred_list_final = []
        for index in range(len(pred_list)):
            prediction = pred_list[index][0]
            pred_list_final.append(np.insert(pred_list[index], 0, (1-prediction)))
        pred_list_final = np.array(pred_list_final)
        return pred_list_final

    # Initialising the LimeTextExplainer
    explainer = LimeTextExplainer(class_names=class_names)

    # Generating random index, explanations are generated for instances at these indexes.
    random_indexes = random.sample(range(0,len(X_test)),10) # <--- Change this value to alter the amount of generated explanations.
    for index in random_indexes: # Iterating over the random_indexes.
        test_text = ' '.join(my_texts[index]) # Joining my_texts strings together at given index.
        exp = explainer.explain_instance(test_text, wrapped_predict, num_features=6, labels=(1,)) # Explaining the instance.
        exp.save_to_file(save_path+'explanation_'+str(index)+'.html') # Saving the explanation as html.
        
        # Implementation to handle the creation and presentation of perturbed data.
        if perturb == True:
            peturbed_string = text_perturbation(my_texts[index]) # Calling function and creating the perturbed data.
            peturbed_string = ' '.join(peturbed_string) # Joining the list of strings.
            exp_peturb = explainer.explain_instance(peturbed_string, wrapped_predict, num_features=6, labels=(1,)) # Explaining the perturbed instance.
            exp_peturb.save_to_file(save_path+'explanation_peturb'+str(index)+'.html') # Saving the perturbed data explanation.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_1 #
############################################
class_names = ['negative', 'positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/lime_text_explanations/text_data_1/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_1(0.2) # Processing the data preparing for training
model_name = "xai_text_classification_data_1_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
lime_text_explainer(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path, perturb=True) # Generating a explanation.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_2 #
############################################
class_names = ['fake', 'real'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/lime_text_explanations/text_data_2/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_2(0.2) # Processing the data preparing for training
model_name = "xai_text_classification_data_2_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
lime_text_explainer(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path, perturb=True) # Generating a explanation.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_3 #
############################################
class_names = ['negative', 'positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/lime_text_explanations/text_data_3/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_3(0.2, 80000) # Processing the data preparing for training
model_name = "xai_text_classification_data_3_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
lime_text_explainer(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path, perturb=True) # Generating a explanation.
