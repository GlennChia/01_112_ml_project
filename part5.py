import numpy as np
from collections import defaultdict

sentence = ['fruit', 'flies', 'like', 'an', 'apple']
tags = ['START', 'A', 'P', 'V', 'D', 'N']
weights = defaultdict(
    lambda: 0,
    {
        ('A', 'A'):-4.00,
        ('an', 'A'): -1.00,
        ('arrow', 'A'): -1.00,
        ('flies', 'A'): -1.00,
        ('like', 'A'): -1.00,
        ('time', 'A'): -1.00,
        ('D', 'N'): 1.00,
        ('an', 'D'): 1.00,
        ('N', 'V'): 1.00,
        ('arrow', 'N'): 1.00,
        ('time', 'N'): 1.00,
        ('P', 'D'): 1.00,
        ('like', 'P'): 1.00,
        ('V', 'P'): 1.00,
        ('flies', 'V'): 1.00,
        ('START', 'A'): -1.00,
        ('START', 'N'): 1.00
    }

)

def get_structured_perceptron_path(tags, sentence, weights):
    """ Get the best path from START to STOP as we go along

    Parameters:
        tags (list): All tags in the dataset
        sentence (list of str): Words for a specific sentence
        weights (defaultdict): Emissions, transtions and their weights

    Returns:
        predicted_sequence(list of tuples): Tuples contain word and tag for that particular index 

    """
    # tags.insert(0,'START')
    tags.pop(0)
    store_scores = np.zeros((len(tags), len(sentence)))
    predicted_sequence = []
    for index_words, word in enumerate(sentence):
        for index_tags, tag in enumerate(tags):
            if index_words == 0:   
                # First word needs to refer to START as the transition to first TAG
                score = weights[('START', tag)] + weights[(word, tag)]
            else:
                # Take the max of the previous layer as the first tag
                previous_layer_max_value = np.max(store_scores, axis=0)[index_words-1]
                # Find the index of the max of the previous layer and get its tag
                previous_layer_max_tag = tags[np.argmax(store_scores,axis=0)[index_words-1]]
                score = previous_layer_max_value + weights[(previous_layer_max_tag, tag)] + weights[(word, tag)]
            store_scores[index_tags][index_words] = score
        current_layer_max_tag = tags[np.argmax(store_scores,axis=0)[index_words]]
        predicted_sequence.append((word, current_layer_max_tag))
    return predicted_sequence

scores = get_structured_perceptron_path(tags, sentence, weights)
print(scores)
'''
Some helpers
For a 2D numpy array, to get the index of the maximum values of each column
i = np.argmax(c,axis=0)

For a 2D numpy array, to get the maximum value of each column
print(np.max(store_scores, axis=0))
'''