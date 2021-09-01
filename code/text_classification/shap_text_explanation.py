# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: shap_text_explanation
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
import os
import os.path
import random

import cntk as C
import numpy as np
import pandas as pd
import shap
from keras.backend.cntk_backend import reverse
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt 

from text_dataset_1_processing import get_dataset_1
from text_dataset_2_processing import get_dataset_2
from text_dataset_3_processing import get_dataset_3

from text_perturbation import text_perturbation

##################
# SHAP EXPLAINER #
##################
def shap_explainer(X_train, X_test, model, word_index, save_path, tokenizer, max_sequence_length, perturb=False):
    
    # We use the first 100 training examples as our background dataset to integrate over
    # Initialising background for DeepExplainer
    background = X_train[:100]
    explainer = shap.DeepExplainer(model, background)

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, word_index.items()))

    # Function takes a tokenized sentence and returns the words of the scentence.
    def sequence_to_text(list_of_indices):
        # Iterating over the tokenized sentence and returning the associated words in a list.
        words = [str(reverse_word_map.get(word)) for word in list_of_indices]
        return words
    my_texts = np.array(list(map(sequence_to_text, X_test))) # "Un-tokenizing" X_test using the function defined above.

    num_explanations = 2 # <--- Change this value to alter the amount of generated explanations.
    
    # Triggering logic if perturbation of instance is wanted.
    if perturb == True:
        # All my_texts[:num_explanations] instances are perturbed.
        X_test_peturbed_words = [list(text_perturbation(X_data)) for X_data in my_texts[:num_explanations]]

        X_test_peturbed_encoded = []
        for lists in X_test_peturbed_words: # Once the instances have been perturbed they need to be retokenized for use with the explainer.
            cnn_rep = tokenizer.texts_to_sequences([lists]) # Tokenizing
            X_test_peturbed_encoded.append(list(pad_sequences(cnn_rep, maxlen=max_sequence_length))) # Applying the original padding amount.
        X_test_peturbed_encoded = np.array([x for y in X_test_peturbed_encoded for x in y])

        # Calculating the shap values of the perturbed instances.
        shap_values_peturbed = explainer.shap_values(X_test_peturbed_encoded[:num_explanations])

    # Calculating the shap values of the original instances.
    shap_values = explainer.shap_values(X_test[:num_explanations])
    
    # Iterating over the amount of explanations we want generated
    for index in range(num_explanations):
        explanation = shap.force_plot(explainer.expected_value[0], shap_values[0][index], my_texts[index], show=False) # Generating a explanation for the original explanation.
        shap.save_html(save_path+'explanation_'+str(index)+'.html', explanation) # Saving the explanation.

        if perturb == True:
            perturbed_explanation = shap.force_plot(explainer.expected_value[0], shap_values_peturbed[0][index], X_test_peturbed_words[index], show=False) # Generating a explanation for the perturbed explanation.
            shap.save_html(save_path+'explanation_peturbed_'+str(index)+'.html', perturbed_explanation) # Saving the explanation. 

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_1 #
############################################
class_names = ['negative', 'positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/shap_text_explanations/text_data_1/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_1(0.33) # Processing the data preparing for training
model_name = "xai_text_classification_data_1_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
shap_explainer(X_train, X_test, model, word_index, save_path, tokenizer, max_sequence_length, perturb=True) # Generating explanations.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_2 #
############################################
class_names = ['fake', 'real'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/shap_text_explanations/text_data_2/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_2(0.33) # Processing the data preparing for training
model_name = "xai_text_classification_data_2_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
shap_explainer(X_train, X_test, model, word_index, save_path, tokenizer, max_sequence_length) # Generating explanations.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_3 #
############################################
class_names = ['negative', 'positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/shap_text_explanations/text_data_3/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_3(0.3, 30000) # Processing the data preparing for training
model_name = "xai_text_classification_data_3_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
shap_explainer(X_train, X_test, model, word_index, save_path, tokenizer, max_sequence_length) # Generating explanations.
