from nltk.tag.perceptron import PerceptronTagger
from preprocessingpart5 import clean_testset, clean_trainset
import nltk
nltk.download('averaged_perceptron_tagger')

tagger = PerceptronTagger()
cleandata = clean_trainset('EN/train')
print(tagger.tag(clean_testset("EN/dev.in", cleandata.smoothed).get_all_sentences()))
