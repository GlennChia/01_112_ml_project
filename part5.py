from collections import defaultdict, Counter
from preprocessingpart5 import clean_trainset, clean_testset
import copy


class StructuredPerceptron:
    def __init__(self, trainfilepath, testfilepath):
        self.weights = defaultdict(int) # initialise a weight dictionary
        # self.goldfeaturevec = defaultdict(int) # initialise a gold standard feature vector
        self.iterations = 5 # number of iterations to run through all the sentences
        self.traindata = clean_trainset(trainfilepath)
        self.testdata = clean_testset(testfilepath, self.traindata)
        self.testfilepath = testfilepath
        self.tags = self.traindata.get_tags()

    def train(self):
        """
        Trains and updates the weight using the structured perceptron algorithm.
        :param traindata: Training data sequences (DATA TYPE TBC)
        :return: weights (dictionary)
        """
        traindata = self.traindata.outputsmootheddata()
        for i in range(self.iterations):
            for sentenceindex in range(len(traindata)):
                actualtagsentence = copy.deepcopy(traindata[sentenceindex])
                actualtagtransitions = [(traindata[sentenceindex][index][1], traindata[sentenceindex][index + 1][1])
                                        for index in range(len(traindata[sentenceindex]) - 1)]

                if sentenceindex == 0:  # for first sentence, randomly initialise tags
                    randomtag = self.tags[0]
                    predtagsentence = [(word, randomtag) for word, tag in traindata[sentenceindex]]
                    predtagtransitions = [(predtagsentence[index][1], predtagsentence[index + 1][1])
                                          for index in range(len(predtagsentence) - 1)]

                    # Get prediction feature vector
                    predfeaturevec = Counter(predtagsentence) + Counter(predtagtransitions)
                    print(predfeaturevec)

                    # Get golden feature vector
                    goldfeaturevec = Counter(actualtagsentence) + Counter(actualtagtransitions)
                    print(goldfeaturevec)

                    # Update weights
                    self.updateweights(predtagsentence, actualtagsentence, predtagtransitions, actualtagtransitions,
                                       predfeaturevec, goldfeaturevec)
                else:
                    # Get the sentence from Viterbi
                    predtagtransitions = [(predtagsentence[index][1], predtagsentence[index + 1][1])
                                          for index in range(len(predtagsentence) - 1)]

                    # Get prediction feature vector
                    predfeaturevec = Counter(predtagsentence) + Counter(predtagtransitions)
                    print(predfeaturevec)

                    # Get golden feature vector
                    goldfeaturevec = Counter(actualtagsentence) + Counter(actualtagtransitions)
                    print(goldfeaturevec)

                    self.updateweights(predtagsentence, actualtagsentence, predtagtransitions, actualtagtransitions,
                               predfeaturevec, goldfeaturevec)


    def predict(self):
        alltestsentences = self.testdata.get_all_sentences()
        predictions = []
        for sentence in alltestsentences:
            #TODO: Run Viterbi and get all the predicted tags
            predictions.append(predictedsentence)

        output_file = self.testfilepath + "/test.p5.out"
        with open(output_file, "w", encoding="utf8") as f:
            for sentence in predictions:
                for predtuple in sentence:
                    word, tag = predtuple
                    f.write(word + " " + tag + "\n")
                f.write("\n")
        f.close()


    def updateweights(self, predtagsentence, actualtagsentence, predtagtransitions, actualtagtransitions,
                      predfeaturevec, goldfeaturevec):
        for tagindex in range(len(predtagsentence)):
            predtag = predtagsentence[tagindex]
            actualtag = actualtagsentence[tagindex]

            if predtag == actualtag:
                continue
            else:
                if actualtag not in predfeaturevec:
                    predval = 0
                else:
                    predval = predfeaturevec[actualtag]
                if predtag not in goldfeaturevec:
                    goldval = 0
                else:
                    goldval = goldfeaturevec[predtag]

                self.weights[predtag] = self.weights[predtag] + (goldval - predfeaturevec[predtag])
                self.weights[actualtag] = self.weights[actualtag] + (goldfeaturevec[actualtag] - predval)

        for tagindex in range(len(predtagtransitions)):
            if predtagtransitions[tagindex] not in actualtagtransitions:
                self.weights[predtagtransitions[tagindex]] = self.weights[predtagtransitions[tagindex]] - \
                                                             predfeaturevec[predtagtransitions[tagindex]]
            if actualtagtransitions[tagindex] not in actualtagtransitions:
                self.weights[actualtagtransitions[tagindex]] = self.weights[predtagtransitions[tagindex]] + \
                                                               goldfeaturevec[actualtagtransitions[tagindex]]


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
