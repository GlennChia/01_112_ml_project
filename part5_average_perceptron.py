import random
from collections import defaultdict
from preprocessingpart5 import clean_trainset, clean_testset
import operator
from copy import deepcopy
from tqdm import tqdm
try:
    import numpy as np
except ImportError:
    pass

random.seed(2)

# ========
# Implementing the data structures
# ========
# Data structure to improve accuracy
high_probability_tags = {}  # When we see this tags we just map
'''
{
    'word1': 'B-NP' 
}
'''




def get_confident_tags(word_tag_sentences, word_occurence, tag_occurence_ratio):
    word_tag_counts = defaultdict(lambda:defaultdict(lambda:0))
    '''
    {
        word: {
            tag1: count1,
            tag2: count2
        }
    }
    '''
    all_tags = set()
    confident_tags = {}
    for word_tag_sentence in word_tag_sentences:
        for word, tag in word_tag_sentence:            
            word_tag_counts[word][tag] += 1
            all_tags.add(tag)
    for word, tag_counts in word_tag_counts.items():
        tag_highest, tag_highest_count  = max(tag_counts.items(), key=lambda a: a[1])
        # tag_highest_count = tag_counts[tag_highest]
        word_frequency = sum(tag_counts.values())
        # Condition check to add confident word-tag pairs
        if word_frequency >= word_occurence and (tag_highest_count/word_frequency) >= tag_occurence_ratio:
            confident_tags[word] = tag_highest
    return confident_tags, all_tags


def train(sentences, confident_tags, all_tags, iterations=11):
    weights = {}  # Temporary placeholder before we compute averaged weights
    weight_totals = {}
    # Data structures for average perceptron
    total_weights_across_updates = defaultdict(lambda:0)  # Crucial for averaging 
    word_tracker = defaultdict(lambda:0)  # Used to update the total-weights at averging step
    update_tracker = 0 # Increments for unconfirmed words
    '''
    {
        'feature_1': {
            'tagA': 3,
            'tagB': -2
        }
    }
    '''
    for iteration in tqdm(range(iterations)):
        sentences_clone = deepcopy(sentences) # preserve original order
        # random.shuffle(sentences_clone)
        for sentence_index in tqdm(range(len(sentences_clone))):
            sentence = sentences_clone[sentence_index]
            # separate the words and sentences 
            words, tags = zip(*sentence)
            # Add START and ENDs to prevent indexign errors later
            format_sentence = ['START-2', 'START'] + list(words) + ['END', 'END+2']
            current_lookback2 = 'START-2'
            current_lookback1 = 'START'
            # Iterate through the words, if they are in confident_tags, pass, else create features, prodict and update
            for word_index, word in enumerate(words):
                training_predictions = confident_tags.get(word)
                # if confident_tag:
                #     pass
                # else:
                if not training_predictions:
                    # Get features for unconfident words
                    i = word_index + 2
                    features = defaultdict(
                        lambda: 0,
                        {
                            'current word {}'.format(word): 1,
                            'previous tag {}'.format(current_lookback1): 1,
                            'previous word {}'.format(format_sentence[i-1]): 1,
                            # Add lookback 2
                            'previous tag2 {}'.format(current_lookback2): 1,
                            'previous word2 {}'.format(format_sentence[i-2]): 1,
                            # combine word combinations
                            #'previous word2 {} previous word {}'.format(format_sentence[i-2], format_sentence[i-1]): 1,
                            #'previous word2 {} previous word {} current word {}'.format(format_sentence[i-2], format_sentence[i-1], word): 1,#
                            #'previous word2 {} current word {}'.format(format_sentence[i-2], word): 1,
                            #'previous word {} current word {}'.format(format_sentence[i-1], word): 1,
                            # Combine tag patterns
                            'previous tag {} current word {}'.format(current_lookback1, word): 1,
                            'previous tag2 {} current word {}'.format(current_lookback2, word): 1,
                            # 'previous tag2 {} previous tag {} current word {}'.format(current_lookback2, current_lookback1, word): 1,
                            'previous tag2 {} previous tag {}'.format(current_lookback2, current_lookback1): 1,
                            # 'previous tag2 {} previous tag {} current word {}'.format(current_lookback2, current_lookback1, word): 1,
                            # Add lookahead
                            'next word {}'.format(format_sentence[i+1]): 1,
                            'next word2 {}'.format(format_sentence[i+2]): 1,
                            # 'next tag {} current word {}'.format(, word): 1,
                            # 'next tag2 {}'.format()
                            # Add prefixes and suffix
                            'current word prefix {}'.format(word[:2]): 1,
                            'current word suffix {}'.format(word[-3:]): 1,
                            # 'previous word prefix {}'.format(format_sentence[i-1][:3]): 1,
                            # 'previous word suffix {}'.format(format_sentence[i-1][-3:]): 1,
                            #'previous word2 prefix {}'.format(current_lookback2[:2]): 1,
                            #'previous word2 suffix {}'.format(current_lookback2[-3:]): 1,
                            'next word prefix {}'.format(format_sentence[i+1][:3]): 1,
                            'next word suffix {}'.format(format_sentence[i+1][-3:]): 1,
                            #'next word2 prefix {}'.format(format_sentence[i+2][:2]): 1,
                            #'next word2 suffix {}'.format(format_sentence[i+2][-3:]): 1,
                            'previous tag prefix {}'.format(current_lookback1[:3]): 1,
                            'previous tag suffix {}'.format(current_lookback1[-3:]): 1,
                            # bias
                            'offset': 1
                        }
                    )
                    
                    # Get guesses 
                    scores = defaultdict(lambda:0.0)
                    for feature, feature_count in features.items():
                        # Iterate through each feature, if present in weights add to score
                        # If not there we skip (If the prediction is wrong we update the weights later similar to structure perceptron)
                        # If there, feature_count will be a 1
                        #print(weights)
                        if feature in weights and feature_count != 0:
                            for tag, tag_count in weights[feature].items():
                                scores[tag] += tag_count #feature_count
                    ### if there is a tie we may be choosing randomly
                    # training_predictions = max(scores.items(), key=lambda a: a[1])
                    try:
                        training_predictions = max(scores.items(), key=lambda a: a[1])[0]
                    except:
                        # We just set to the random tag or we can do highest probability tag
                        training_predictions = random.sample(all_tags, 1)[0]
                    # Update if needed if the tags for a word don't match
                    update_tracker += 1
                    if training_predictions != tags[word_index]:
                        for feature in features:
                            temp_weights = weights.setdefault(feature, {})
                            # Similar to structured, we add 1 to features for the correct
                            feature_tag_pair = (feature, tags[word_index])
                            indiv_temp_weight = temp_weights.get(tags[word_index], 0.0)
                            total_weights_across_updates[feature_tag_pair] += indiv_temp_weight * (update_tracker - word_tracker[feature_tag_pair])  # Crucial for averaging 
                            word_tracker[feature_tag_pair] = update_tracker
                            weights[feature][tags[word_index]] = indiv_temp_weight + 1.0
                            # Similar to structured we sub 1 from features for incorrect
                            feature_tag_pair = (feature, training_predictions)
                            indiv_temp_weight = temp_weights.get(training_predictions, 0.0)
                            total_weights_across_updates[feature_tag_pair] += indiv_temp_weight * (update_tracker - word_tracker[feature_tag_pair])  # Crucial for averaging 
                            word_tracker[feature_tag_pair] = update_tracker
                            weights[feature][training_predictions] = indiv_temp_weight - 1.0
                current_lookback2 = current_lookback1
                current_lookback1 = training_predictions
        random.shuffle(sentences_clone)
    print('Averaging weights')
    # average the features
    for feature, tag_weights in weights.items():
        averaged_feature_weights = {}
        for tag, tag_weight in tag_weights.items():
            feature_tag_pair = (feature, tag)
            # Account for the last update 
            if update_tracker ==word_tracker[feature_tag_pair]:
                total = total_weights_across_updates[feature_tag_pair] + tag_weight
            else:
                total = total_weights_across_updates[feature_tag_pair]
                total += (update_tracker - word_tracker[feature_tag_pair]) * tag_weight
            averaged = round(total / update_tracker, 4)
            if averaged:
                averaged_feature_weights[tag] = averaged
        weights[feature] = averaged_feature_weights
    return weights

def predict(sentences, confident_tags, all_tags, weights):
    predsentences = []
    predsentence = []
    for sentence_index in tqdm(range(len(sentences))):
        sentence = sentences[sentence_index]
        # Add START and ENDs to prevent indexign errors later
        format_sentence = ['START-2', 'START'] + sentence + ['END', 'END+2']
        current_lookback2 = 'START-2'
        current_lookback1 = 'START'
        # Create arrays to store predictions
        # Iterate through the words, if they are in confident_tags, pass, else create features, prodict and update
        for word_index, word in enumerate(sentence):
            training_predictions = confident_tags.get(word)
            # if confident_tag:
            #     pass
            # else:
            if not training_predictions:
                # Get features for unconfident words
                i = word_index + 2
                features = defaultdict(
                    lambda: 0,
                    {
                        'current word {}'.format(word): 1,
                        'previous tag {}'.format(current_lookback1): 1,
                        'previous word {}'.format(format_sentence[i-1]): 1,
                        # Add lookback 2
                        'previous tag2 {}'.format(current_lookback2): 1,
                        'previous word2 {}'.format(format_sentence[i-2]): 1,
                        # combine word combinations
                        #'previous word2 {} previous word {}'.format(format_sentence[i-2], format_sentence[i-1]): 1,
                        #'previous word2 {} previous word {} current word {}'.format(format_sentence[i-2], format_sentence[i-1], word): 1,#
                        #'previous word2 {} current word {}'.format(format_sentence[i-2], word): 1,
                        #'previous word {} current word {}'.format(format_sentence[i-1], word): 1,
                        # Combine tag patterns
                        'previous tag {} current word {}'.format(current_lookback1, word): 1,
                        'previous tag2 {} current word {}'.format(current_lookback2, word): 1,
                        # 'previous tag2 {} previous tag {} current word {}'.format(current_lookback2, current_lookback1, word): 1,
                        'previous tag2 {} previous tag {}'.format(current_lookback2, current_lookback1): 1,
                        # 'previous tag2 {} previous tag {} current word {}'.format(current_lookback2, current_lookback1, word): 1,
                        # Add lookahead
                        'next word {}'.format(format_sentence[i+1]): 1,
                        'next word2 {}'.format(format_sentence[i+2]): 1,
                        # 'next tag {} current word {}'.format(, word): 1,
                        # 'next tag2 {}'.format()
                        # Add prefixes and suffix
                        'current word prefix {}'.format(word[:2]): 1,
                        'current word suffix {}'.format(word[-3:]): 1,
                        # 'previous word prefix {}'.format(format_sentence[i-1][:3]): 1,
                        # 'previous word suffix {}'.format(format_sentence[i-1][-3:]): 1,
                        #'previous word2 prefix {}'.format(current_lookback2[:2]): 1,
                        #'previous word2 suffix {}'.format(current_lookback2[-3:]): 1,
                        'next word prefix {}'.format(format_sentence[i+1][:3]): 1,
                        'next word suffix {}'.format(format_sentence[i+1][-3:]): 1,
                        #'next word2 prefix {}'.format(format_sentence[i+2][:2]): 1,
                        #'next word2 suffix {}'.format(format_sentence[i+2][-3:]): 1,
                        'previous tag prefix {}'.format(current_lookback1[:3]): 1,
                        'previous tag suffix {}'.format(current_lookback1[-3:]): 1,
                        # bias
                        'offset': 1
                    }
                )
                
                # Get guesses 
                scores = defaultdict(lambda:0.0)
                for feature, feature_count in features.items():
                    # Iterate through each feature, if present in weights add to score
                    # If not there we skip (If the prediction is wrong we update the weights later similar to structure perceptron)
                    # If there, feature_count will be a 1
                    #print(weights)
                    if feature in weights and feature_count != 0:
                        for tag, tag_count in weights[feature].items():
                            scores[tag] += tag_count #feature_count
                ### if there is a tie we may be choosing randomly
                # training_predictions = max(scores.items(), key=lambda a: a[1])
                try:
                    training_predictions = max(scores.items(), key=lambda a: a[1])[0]
                except:
                    # We just set to the random tag or we can do highest probability tag
                    training_predictions = random.sample(all_tags, 1)[0]
            predsentence.append((word, training_predictions))
            current_lookback2 = current_lookback1
            current_lookback1 = training_predictions
        predsentences.append(predsentence)
        predsentence = []

    return predsentences


if __name__ == '__main__':
    '''
    python .\EvalScript\evalResult.py EN/dev.out EN/dev.p5.out
    python .\EvalScript\evalResult.py AL/dev.out AL/dev.p5.out
    '''
    # =======
    # Hyper parameters
    # =======
    iterations = 11
    # For confident predictions
    word_occurence = 19
    tag_occurence_ratio = 0.98

    def run_predictions(folder, part):
        # =======
        # Data pre-processing
        # =======
        # Smooth the train and the test
        cleantrain = clean_trainset('{}/train'.format(folder))
        cleantest = clean_testset('{}/{}.in'.format(folder, part), cleantrain.smoothed)
        # Format tarin data into [[('',START), ('word1', 'tagA'), ..., ('wordn', 'tagB')] ...]
        cleantrain_format = cleantrain.outputsmootheddata()
        # Format test data into [[word1, word2, ..., wordn], ...]
        cleantest_format = cleantest.get_all_sentences()
        # Getting a dictionary of confident tags and a set of all tags
        confident_tags, all_tags = get_confident_tags(cleantrain_format, word_occurence, tag_occurence_ratio)
        #print(confident_tags)
        print('BEGIN TRAINING FOR {} PART {}'.format(folder, part))
        final_weights = train(cleantrain_format, confident_tags, all_tags, iterations)
        #print(final)
        print('BEGIN PREDICTIONS')
        predicted_results = predict(cleantest_format, confident_tags, all_tags, final_weights)
        output_file = "{}/{}.p5.out".format(folder, part)
        with open(output_file, "w", encoding="utf8") as f:
            for sentence in predicted_results:
                for predtuple in sentence:
                    word, tag = predtuple
                    f.write(word + " " + tag + "\n")
                f.write("\n")
        f.close()

    # TO RUN THE CODE FOR PART 5a
    run_predictions('EN', 'dev')
    run_predictions('AL', 'dev')
    # TO RUN THE CODE FOR PART 5b
    run_predictions('EN', 'test')
    run_predictions('AL', 'test')