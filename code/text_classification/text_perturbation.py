# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: text_perturbation
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
from nltk.corpus import wordnet
import random

def text_perturbation(strings):
    changed_words = []

    # Selecting 15% of the indexes of the instance randomly/
    random_indexes_strings = random.sample(range(1,len(strings)),int(len(strings)*0.15))

    # Random selection if antonyms or synonyms should be used as replacements.
    synonyms_or_antonyms = random.randint(0,1)

    # Iterating over the radom indexes calculated in line 16
    for index in random_indexes_strings:
        
        # If synonym implememtation
        if synonyms_or_antonyms == 0:
            # Loops through and adds all synonyms for the current index into list.
            for syn in wordnet.synsets(str(strings[index])):
                for l in syn.lemmas():
                    if len(l.name()) > 0:
                        changed_words.append(l.name())
            # Making the list of synonyms into a set to account for any possible duplicates
            words_finished = list(set(changed_words))

        else:
            # Loops through and adds all antonyms for the current index into list.
            for syn in wordnet.synsets(str(strings[index])):
                for l in syn.lemmas():
                    if len(l.antonyms()) > 0:
                        changed_words.append(l.antonyms()[0].name())
            # Making the list of antonyms into a set to account for any possible duplicates
            words_finished = list(set(changed_words))

        # Checks to see that there are synonyms or antonyms.
        if len(words_finished) > 0:
            # Randomly selects a synonym or antonym for the index from the list.
            random_index_synonyms = random.randrange(0, len(words_finished))
            # Adds the chosen synonym or antonym to the final list at the correct index.
            strings[index] = str(words_finished[random_index_synonyms])
        
        # Clearing for next loop.
        changed_words.clear()
        words_finished.clear()

    # Returning the perturbed instance.
    return strings
        
            
        
