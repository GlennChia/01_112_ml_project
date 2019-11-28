import numpy as np
import pandas as pd
import itertools
import copy

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


def replacewordtrain(word, word_counts, k):
    if word_counts[word] < k:
        return "#UNK#"
    return word


def smoothingtrain(data, k=3):
    word_counts = data['words'].value_counts().to_dict()
    data['words'] = data['words'].apply(lambda word: replacewordtrain(word, word_counts, k))
    return data

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

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))

class node_k():
    def __init__(self, t, n, k):
        self.t = t
        self.n = n
        self.k = k
        self.score = 0
        self.parent = ""
        self.subpath = ""

    def __repr__(self):
        return ("node({}, {}, {})".format(self.t, self.n, self.k))

class viterbi():
    def __init__(self, emission, transition, sentence, tags):
        """
        :param emission: lookup of emission probabilities (dict) (tag, word) --> probability
        :param transition: lookup of transmission probbilities (dict) (tag1, tag2) --> probability
        :param sentence: string sentence
        :param tags: array of tags available for dataset
        """
        self.emission = emission
        self.transition = transition
        self.sentence = sentence.split(" ")
        self.n = len(sentence) + 2
        self.t = tags
        self.tree_k, self.finalscore_k, self_kpaths = self.populate_tree_2(2)

    def build_tree_2(self, k):
        """
        t = # of tags available
        n = length of sentence
        k = choice of how many paths
        :return:
        """
        return np.zeros((len(self.sentence), len(self.t), k))

    def get_transition_array(self, dest_node):
        out = np.zeros((len(self.t), 1))
        idx = 0
        for j in self.t:
            print(j, dest_node)
            if (j, dest_node) in self.transition:
                out[idx] = self.transition[j, dest_node]
            idx += 1

        print("transition array: ", out)
        return out



    def populate_tree_2(self, k_best):
        """
        Returns populated tree with k-best scores for each node
        :param k: choice of k-best scores
        :return: np.array of size (n, t, k). see build_tree_2 for more details.
        """
        tree = self.build_tree_2(k_best)
        all_nodes = [[[node_k(i, j,  k) for k in range(k_best)]for j in range(len(self.t))] for i in range(len(self.sentence))]

        for i in range(len(self.sentence) + 1):
            if i == 0:
                idx = 0
                for j in self.t:
                    if ("START", j) not in self.transition or (j, self.sentence[0]) not in self.emission:
                        tree[i, idx, 0] = 0
                        # all_nodes[idx][i][0].score = 0
                    else:
                        tree[i, idx, 0] = self.transition[("START", j)] * self.emission[(j, self.sentence[0])]
                        # all_nodes[idx][i][0] = self.transition[("START", j)] * self.emission[(j, self.sentence[0])]
                    idx += 1


            elif i == len(self.sentence):
                last_layer = []

                raw_scores = tree[i-1]  # extracts column (k x 1) scores in i-1
                candidate_scores = raw_scores * self.get_transition_array("STOP")
                max_ind = k_largest_index_argsort(candidate_scores, k_best)
                final_score = [candidate_scores[tuple(idx)] for idx in max_ind]


                for l in range(k_best):
                    last_node = node_k(i, 0, 0)
                    j_coord = max_ind[l][0]
                    k_coord = max_ind[l][1]

                    last_node.parent = all_nodes[i - 1][j_coord][k_coord]
                    last_node.subpath = self.t[j_coord]
                    last_layer.append(last_node)

            else:
                raw_scores = tree[i-1]  # extracts column (k x 1) scores in i-1
                print(" raw scores: ", raw_scores)

                idx_k = 0
                for k in self.t:
                    print("in label: ", k)
                    candidate_scores = raw_scores * self.get_transition_array(k)
                    print("candidate scores: ", candidate_scores)
                    max_ind = k_largest_index_argsort(candidate_scores, k_best)

                    if (k, self.sentence[i]) not in self.emission:
                        tree[i, idx_k] = 0
                    else:
                        tree[i, idx_k] = np.array([candidate_scores[tuple(idx)] for idx in max_ind]) * self.emission[k, self.sentence[i]]

                    for l in range(k_best):
                        j_coord = max_ind[l][0]
                        k_coord = max_ind[l][1]

                        current_node = all_nodes[i][idx_k][l]
                        current_node.parent = all_nodes[i-1][j_coord][k_coord]
                        current_node.subpath = self.t[j_coord]
                    idx_k += 1

        k_paths = []
        for a in range(k_best):
            path = []
            current_node = last_layer[a]
            for i in range(len(self.sentence), 0, -1):
                path.insert(0, current_node.subpath)
                current_node = current_node.parent

            k_paths.append(path)

        return tree, final_score, k_paths

transition_lookup = {("START", "A"): 1.0, ("A", "A"): 0.5 , ("A", "B"): 0.5, ("B", "B"): 0.8, ("B", "STOP"): 0.2}
emission_lookup = {("A", "the"): 0.9, ("A", "dog"): 0.1, ("B", "the"): 0.1, ("B", "dog"): 0.9}

sentence = "the dog the"
# print(sentence.split(" "))
test = viterbi(emission_lookup, transition_lookup, sentence, ["A", "B"])
print(test.populate_tree_2(2))
# print(test.path)

print(("A", "A") in transition_lookup)