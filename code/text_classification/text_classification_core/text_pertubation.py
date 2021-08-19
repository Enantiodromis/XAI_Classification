from nltk.corpus import wordnet
import random

def find_synonyms_and_perturb(strings):
    synonyms = []
    random_indexes_strings = random.sample(range(1,len(strings)),int(len(strings)*0.2))
    #random_indexes_strings = random.sample(range(1,len(strings)),2)
    for index in random_indexes_strings:
        print(index)
        print(strings[index])
        for syn in wordnet.synsets(str(strings[index])):
            for lm in syn.lemmas():
                synonyms.append(lm.name())
        synonyms_finished = list(set(synonyms))
        if len(synonyms_finished) > 0:
            random_index_synonyms = random.randrange(0, len(synonyms_finished))
            strings[index] = str(synonyms_finished[random_index_synonyms])
        print(strings[index])
        synonyms.clear()
        synonyms_finished.clear()
    return strings
        
            
        
