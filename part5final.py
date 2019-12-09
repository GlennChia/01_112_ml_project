from collections import defaultdict, Counter
from preprocessingpart5 import clean_trainset, clean_testset
import copy
import numpy as np


class StructuredPerceptron:
    def __init__(self, trainfilepath):
        self.weights = defaultdict(int) # initialise a weight dictionary
        # self.goldfeaturevec = defaultdict(int) # initialise a gold standard feature vector
        self.iterations = 5 # number of iterations to run through all the sentences
        self.traindata = clean_trainset(trainfilepath)
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

                # if sentenceindex == 0:  # for first sentence, randomly initialise tags
                #     randomtag = self.tags[0]
                #     predtagsentence = []
                #     for word, tag in traindata[sentenceindex]:
                #         if tag == 'START':
                #             predtagsentence.append((word, tag))
                #         else:
                #             predtagsentence.append((word, randomtag))
                #
                #     predtagtransitions = [(predtagsentence[index][1], predtagsentence[index + 1][1])
                #                           for index in range(len(predtagsentence) - 1)]
                #
                #     # Get prediction feature vector
                #     predfeaturevec = Counter(predtagsentence) + Counter(predtagtransitions)
                #
                #     # Get golden feature vector
                #     goldfeaturevec = Counter(actualtagsentence) + Counter(actualtagtransitions)
                #
                #     # Update weights
                #     self.updateweights(predtagsentence, actualtagsentence, predtagtransitions, actualtagtransitions,
                #                        predfeaturevec, goldfeaturevec)
                # else:
                sentencewithouttags = []
                for word, tag in traindata[sentenceindex]:
                    if tag != 'START':
                        sentencewithouttags.append(word)
                predtagsentence = self.get_structured_perceptron_path(sentencewithouttags)
                # Get the sentence from Viterbi
                predtagtransitions = [(predtagsentence[index][1], predtagsentence[index + 1][1])
                                      for index in range(len(predtagsentence) - 1)]

                # Get prediction feature vector
                predfeaturevec = Counter(predtagsentence) + Counter(predtagtransitions)

                # Get golden feature vector
                goldfeaturevec = Counter(actualtagsentence) + Counter(actualtagtransitions)

                self.updateweights(predtagsentence, actualtagsentence, predtagtransitions, actualtagtransitions,
                                   predfeaturevec, goldfeaturevec)


    def predict(self, testfilepath):
        testdata = clean_testset(testfilepath + '/dev.in', self.traindata.smoothed)
        alltestsentences = testdata.get_all_sentences()
        predictions = []
        for sentence in alltestsentences:
            predictedsentence = self.get_structured_perceptron_path(sentence)
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
                      predfeaturevec, goldfeaturevec, learning_rate=0.2):
        for tag in goldfeaturevec.keys():
            if tag in predfeaturevec.keys():
                self.weights[tag] = self.weights[tag] + (goldfeaturevec[tag] - predfeaturevec[tag])
            else:
                self.weights[tag] = self.weights[tag] + goldfeaturevec[tag]

        for tag in predfeaturevec.keys():
            if tag in goldfeaturevec.keys():
                # pass, because we already updated the weights
                continue
            else:
                self.weights[tag] = self.weights[tag] - predfeaturevec[tag]



    def get_structured_perceptron_path(self, sentence):
        """ Get the best path from START to STOP as we go along

        Parameters:
            tags (list): All tags in the dataset
            sentence (list of str): Words for a specific sentence
            weights (defaultdict): Emissions, transtions and their weights

        Returns:
            predicted_sequence(list of tuples): Tuples contain word and tag for that particular index

        """
        # tags.insert(0,'START')
        tags = copy.deepcopy(self.tags)
        tags.pop(0)
        store_scores = np.zeros((len(tags), len(sentence)))
        predicted_sequence = []
        for index_words, word in enumerate(sentence):
            for index_tags, tag in enumerate(tags):
                if index_words == 0:
                    # First word needs to refer to START as the transition to first TAG
                    score = self.weights[('START', tag)] + self.weights[(word, tag)]
                else:
                    # Take the max of the previous layer as the first tag
                    previous_layer_max_value = np.max(store_scores, axis=0)[index_words - 1]
                    # Find the index of the max of the previous layer and get its tag
                    previous_layer_max_tag = tags[np.argmax(store_scores, axis=0)[index_words - 1]]
                    score = previous_layer_max_value + self.weights[(previous_layer_max_tag, tag)] + self.weights[(word, tag)]
                store_scores[index_tags][index_words] = score
            current_layer_max_tag = tags[np.argmax(store_scores, axis=0)[index_words]]
            predicted_sequence.append((word, current_layer_max_tag))
        return predicted_sequence


trainfilepath = 'EN/train'
testfilepath = 'EN'
perceptron = StructuredPerceptron(trainfilepath)
perceptron.train()
perceptron.predict(testfilepath)
