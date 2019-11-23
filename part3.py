import pandas as pd
import numpy as np
from copy import deepcopy
import json

def pretty(some_dict):
    return json.dumps(some_dict, sort_keys=True, indent=4)


def read_to_pdf(file_path):
    with open(file_path, encoding='utf8') as f_message:
        temp = f_message.read().splitlines()
    words = []
    tags = []
    for index, word_tags in enumerate(temp):
        if index == 0:
            words.append('')
            tags.append('START')
        elif index == len(temp) - 1:
            words.append('')
            tags.append('STOP')
        elif word_tags == '':
            words.append('')
            tags.append('STOP')
            words.append('')
            tags.append('START')
        else:
            split_word_tags = word_tags.split(' ')
            words.append(split_word_tags[0])
            tags.append(split_word_tags[1])
    tags_next = tags[1:]
    df = pd.DataFrame(list(zip(words, tags, tags_next)), columns =['words', 'tags', 'tags_next']) 
    df['tags_next'] = df['tags_next'].str.replace('START', '')
    return df


def read_to_pdf_test(file_path):
    with open(file_path, encoding="utf8") as f_message:
        temp = f_message.read().splitlines()
        temp = list(filter(None, temp))
    df = pd.DataFrame(temp, columns=['words'])
    return df


def estimate_transition_parameters(df):
    """Return a dataframe with 
    tag | next_tag | count_tag | count_transition | transition_prob

    Parameters:
    df (DataFrame): Dataframe with word, tags, tags_next

    Returns:
    df (DataFrame)
    """
    df['count_tag'] = df.groupby(['tags']).tags.transform(np.size)
    df['count_transition'] = df.groupby(['tags', 'tags_next']).tags.transform(np.size)
    df.loc[df.tags_next == '', 'count_transition'] = 0
    df['transition_probability'] = df['count_transition'] / df['count_tag']
    df = df.drop_duplicates(subset=['tags', 'tags_next'])
    df = df.drop(['words'], axis=1)
    df = df.sort_values(['tags','tags_next'])
    df = df.reset_index()
    df = df.drop(['index'], axis=1)
    return df


def estimate_emission_parameters(df):
    """
    Calculates the emission probabilities from count of words/ count of tags
    :param df: raw word to tag map
    :return: columns = count of word, count of tags, all emission probabilites of tag --> word
    """
    count_emit = df.groupby(['tags', 'words']).size().reset_index()
    count_emit.columns =["tags", "words", "count_emit"]
    count_tags = df.groupby(["tags"]).size().reset_index()
    count_tags.columns = ["tags", "count_tags"]

    count = pd.merge(count_emit, count_tags, on="tags")
    count["emission"] = count["count_emit"]/count["count_tags"]

    return count.drop(columns=["count_emit", "count_tags"])


def replacewordtrain(word, word_counts, k):
    if word_counts[word] < k:
        return "#UNK#"
    return word

def replacewordtest(word, train):
    if word in train:
        return word
    return "#UNK#"


def smoothingtrain(data, k=3):
    word_counts = data['words'].value_counts().to_dict()
    data['words'] = data['words'].apply(lambda word: replacewordtrain(word, word_counts, k))
    return data

def smoothingtest(testdata, traindata):
    trainvalues = set(traindata['words'])
    testdata['words'] = testdata['words'].apply(lambda word: replacewordtest(word, trainvalues))
    return testdata


class Node:
    def __init__(self, state, weight=0):
        self._state = state
        # self._transition = transition
        # self._emission = emission
        self._weight = weight

class ViterbiTree:
    def __init__(self, layers):
        self.layers = [[] for i in range(layers)]

def viterbi_algorithm(data_in, transition_full, emission_full):
    number_of_inner_layers = len(data_in) # change this when we work with more examples
    # viterbi_tree = ViterbiTree(number_of_inner_layers + 2)
    '''
{
    0: {
        start: 1
    },
    1: {
        'B-ADJP': 0.2,
        'o': 0.2,
        x_val: ''
    },
    2:{
        '': 0.9,
        x_val: ''
    },
    3:{
        stop: 0.2
    }
}
    '''
    states = transition_full.tags.unique().tolist()
    states.remove('STOP')
    states.remove('START')
    number_of_states = len(states)
    viterbi_tree = {}
    for i in range(number_of_inner_layers + 2):
        if i == 0:
            viterbi_tree[i] = {
                "START": 1
            }
        elif i == number_of_inner_layers + 1:
            viterbi_tree[i] = {
                "STOP": 0
            }
        else:
            indiv_dict = {}
            for index, state in enumerate(states):
                indiv_dict[state] = 0
            clone_dict = deepcopy(indiv_dict)
            viterbi_tree[i] = clone_dict
            viterbi_tree[i]['x_val'] = data_in.iloc[i-1]['words']
    # Compute updated weights
    # iterate through each layer
    for j in range(number_of_inner_layers):
        # iterate thorugh each state
        for state in viterbi_tree[j+1].keys():
            if state != "x_val":
                # get emission
                emission = emission_full.loc[(emission_full['tags'] == state) & (emission_full['words'] == viterbi_tree[j+1]["x_val"]), 'emission']#.iloc[0] 
                try:
                    emission = emission.iloc[0]
                except:
                    emission = 0.0
                score_array = []
                for prev_state, prev_state_value in viterbi_tree[j].items():
                    if prev_state != 'x_val':
                        #compute transition
                        transition = transition_full.loc[(transition_full['tags'] == prev_state) & (transition_full['tags_next'] == state), 'transition_probability']#.iloc[0]
                        try:
                            transition = transition.iloc[0]
                        except:
                            transition = 0.0
                        # compute score
                        score_array.append(prev_state_value * emission * transition)
                viterbi_tree[j+1][state] = max(score_array)
    # Find the state sequence
    output_sequence = ['STOP']
    for k in range(number_of_inner_layers, 0, -1):
        # Iterate thorugh each state
        highest_score = 0
        for state, value in viterbi_tree[k].items():
            if state != 'x_val':
                # Find transition
                transition = transition_full.loc[(transition_full['tags'] == state) & (transition_full['tags_next'] == output_sequence[-1]), 'transition_probability']#.iloc[0]
                try:
                    transition = transition.iloc[0]
                except:
                    transition = 0.0
                # compute score
                score = value * transition
                if score > highest_score:
                    highest_score = score
                    chosen_state = state
        output_sequence.append(chosen_state)
    output_sequence.remove('STOP')
    output_sequence.reverse()
    data_in['predicted_states'] = output_sequence
    return data_in


if __name__=="__main__":
    '''Part 3 Qn 1: Test transition parameters'''
    en_path = 'EN/train'
    sg_path = 'SG/train'
    al_path = 'AL/train'
    cn_path = 'CN/train'
    df_en = read_to_pdf(en_path)
    df_en = smoothingtrain(df_en)
    #print(df_en)
    df_transition = estimate_transition_parameters(df_en)
    # print(df_transition)
    df_emission = estimate_emission_parameters(df_en)
    #print(df_emission)
    df_test = read_to_pdf_test('EN/dev copy.in')
    df_test = smoothingtest(df_test, df_en)
    #print(df_emission[df_emission['words'] == 'HBO'])
    #print(df_test)
    print(viterbi_algorithm(df_test, df_transition, df_emission))
    # df_sg = read_to_pdf(sg_path)
    # print(df_sg)
    # print(estimate_transition_parameters(df_sg))

