# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: anchor_text_explanation
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
import os
import os.path
import random

import numpy as np
import spacy
from anchor.anchor_text import AnchorText
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

from text_dataset_1_processing import get_dataset_1
from text_dataset_2_processing import get_dataset_2
from text_dataset_3_processing import get_dataset_3

from text_perturbation import text_perturbation

####################################
# ANCHOR TEXT EXPLANATION FUNCTION #
####################################
def extracting_anchors_explanation(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path, perturb=False):
    
    # Creating a reverse word dictionary
    reverse_word_map = dict(map(reversed, word_index.items()))
    
    # Function takes a tokenized sentence and returns the words of the scentence.
    def sequence_to_text(list_of_indices):
        # Iterating over the tokenized sentence and returning the associated words in a list.
        words = [str(reverse_word_map.get(word)) for word in list_of_indices]
        return words
    my_texts = np.array(list(map(sequence_to_text, X_test))) # "Un-tokenizing" X_test using the function defined above.

    # Wrapped prediction function returns 0 or 1 depending on returned prediction from model. 
    # As the model outputs a predicition in the form [prediction]. The prediction is a value between 0 - 1
    # If the prediction is < 0.5 it is associated with the class at index 0, else it is the class at index 1.
    def wrapped_predict(strings):
        cnn_rep = tokenizer.texts_to_sequences(strings)
        text_data = pad_sequences(cnn_rep, maxlen=max_sequence_length)
        prediction = model.predict(text_data)
        predicted_class = np.where(prediction > 0.5, 1,0)[0]
        return predicted_class

    # Loading in the NLP spacy model used to create perturbations of the inputted instance.
    nlp = spacy.load('en_core_web_sm')
    # Initialising the AnchorText explainer.
    explainer = AnchorText(nlp, class_names, use_unk_distribution=True)

    # Generating random index, explanations are generated for instances at these indexes.
    random_indexes = random.sample(range(0,len(X_test)),10) # <--- Change this value to alter the amount of generated explanations.
    for index in random_indexes: # Iterating through the random_indexes.
        test_text = ' '.join(my_texts[index]) # Joining the "un-tokenized" X_test which is in the form [[word], [word], [word], [word]]
        exp = explainer.explain_instance(test_text, wrapped_predict, threshold=0.95) # Explaining the instance
        exp.save_to_file(save_path+'explanation_'+str(index)+'.html') # Saving the explanation as a HTML file

        if perturb == True:
            peturbed_string = text_perturbation(my_texts[index]) # Calling function and creating the perturbed data.
            peturbed_string = ' '.join(peturbed_string) # Joining the list of strings.
            exp_perturbed = explainer.explain_instance(test_text, wrapped_predict, threshold=0.95) # Explaining the perturbed instance
            exp_perturbed.save_to_file(save_path+'explanation_perturbed'+str(index)+'.html') # Saving the explanation as a HTML file

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_1 #
############################################
class_names = ['negative', 'positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/anchors_text_explanations/text_data_1/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_1(0.2) # Processing the data preparing for training
model_name = "xai_text_classification_data_1_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
extracting_anchors_explanation(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path, perturb=True) # Generating explanations.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_2 #
############################################
class_names = ['fake', 'real'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/anchors_text_explanations/text_data_2/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_2(0.2) # Processing the data preparing for training
model_name = "xai_text_classification_data_2_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
extracting_anchors_explanation(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path, perturb=True) # Generating explanations.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_3 #
############################################
class_names = ['negative', 'positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/anchors_text_explanations/text_data_3/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_3(0.2, 80000) # Processing the data preparing for training
model_name = "xai_text_classification_data_3_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
extracting_anchors_explanation(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path, perturb=True) # Generating explanations.
