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


