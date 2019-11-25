import numpy as np
import pandas as pd
import itertools

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

class node():
    def __init__(self, t, n):
        self.t = t
        self.n = n
        self.score = 0
        self.parent = ""
        self.tag = ""

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
        self.tree, self.maxscore = self.populate_tree()
        self.tree_k, self.finalscore_k = self.populate_tree_2(2)

    def get_score(self, n, t):
        """
        calculates score of a particular node
        :param n: # of words in sentence + 2 (START, STOP)
        :param t: # of states
        :return: array of max 7 scores scores
        """
        if n == 0:
            print("AT START NODE")
            return 1
        elif n == 1:
            print("AT FIRST LAYER")
            calc_scores = []
            for i in self.t:
                if ("START", i) not in self.transition or (i, self.sentence[n-1]) not in self.emission:
                    continue
                else:
                    score = self.get_score(0, 0) * self.transition[("START", i)] * self.emission[(i, self.sentence[n-1])]
                    calc_scores.append(score)
                    # calc_scores.sort()
                    print(score)
                return max(calc_scores)

        elif (n == self.n):
            print("AT LAST NODE")
            calc_score = []
            for i in range(len(self.t)):
                if (self.t[i], "STOP") not in self.transition:
                    continue
                score = self.get_score(n-1, 1) * self.transition[(self.t[i], "STOP")]
                calc_score.append(score)
            return max(calc_score)
        else:
            print("IN MIDDLE LAYER")
            calc_score = []
            for i in range(len(self.t)):
                if (self.t[i], self.t[t]) not in self.transition or (self.t[i], self.sentence[n-1]) not in self.emission:
                    continue
                score = self.get_score(n-1, i) * self.transition[(self.t[i], self.t[t])] * self.emission[(self.t[i], self.sentence[n-1])]
                calc_score.append(score)
            calc_score.sort()
            return max(calc_score)

    def build_tree(self):
        """
        initialise viterbi tree (middle layers)
        :return: np.array of size = (t, n); t= rows, n = length
        """
        return np.zeros((len(self.t), len(self.sentence)))


    def tree_layer1(self):
        init = np.zeros(len(self.t))
        idx = 0
        for i in self.t:
            if ("START", i) not in self.transition or (i, self.sentence[0]) not in self.emission:
                init[idx] = 0.0
            else:
                init[idx] = self.transition[("START", i)] * self.emission[(i, self.sentence[0])]
            idx += 1
        return init

    def populate_tree(self):
        """
        Calculate all scores in viterbi tree, except final score
        :return: np.array of max scores of each node, maxscore of path
        """
        tree = self.build_tree()
        all_nodes = [[node(j, i) for j in range(len(self.t))] for i in range(len(self.sentence))]

        for i in range(len(self.sentence)+1):
            if i == 0:
                idx = 0
                for j in self.t:
                    if ("START", j) not in self.transition or (j, self.sentence[0]) not in self.emission:
                        tree[idx, i] = 0.0
                    else:
                        tree[idx, i] = self.transition[("START", j)] * self.emission[(j, self.sentence[0])]
                    idx += 1

            elif i == len(self.sentence):
                all_scores = []
                idx = 0
                for j in self.t:
                    if (j, "STOP") not in self.transition:
                        all_scores.append(0)
                    else:
                        score = tree[idx, -1] * self.transition[(j, "STOP")]
                        all_scores.append(score)
                    idx += 1
                maxscore = max(all_scores)
                tag_idx = all_scores.index(max(all_scores))
                tag = self.t[tag_idx]
                # all_nodes[0][i] = node(0, i)
                last_node = node(0, i)
                last_node.parent = all_nodes[tag_idx][-1]
                last_node.tag = tag

            else:
                idx = 0
                for j in self.t:
                    all_scores = []
                    idx_k = 0
                    for k in self.t:
                        print("Inner: " + k)
                        if (k, j) not in self.transition or (j, self.sentence[i]) not in self.emission:
                            all_scores.append(0)
                        else:
                            score = tree[idx_k, i-1] * self.transition[(k, j)] * self.emission[(j, self.sentence[i])]
                            all_scores.append(score)
                        idx_k += 1
                    tree[idx, i] = max(all_scores)

                    tag_idx = all_scores.index(max(all_scores))
                    tag = self.t[tag_idx]
                    # all_nodes[idx][i] = node(idx, i)
                    all_nodes[idx][i-1].tag = tag
                    if i != 1:
                        all_nodes[idx][i-1].parent = all_nodes[tag_idx][i-2]
                    else:
                        pass

                    idx += 1

        path = []
        current_node = last_node
        for i in range(len(self.sentence), 0,  -1):
            path.insert(0, current_node.tag)
            current_node = current_node.parent
        print(path)

        return tree, maxscore



    def build_tree_2(self, k):
        """
        n = length of sentence
        t = # of tags available
        k = choice of how many paths
        :return:
        """
        return np.zeros((len(self.t), len(self.sentence), k))

    def populate_tree_2(self, k_best):
        """
        Returns populated tree with k-best scores for each node
        :param k: choice of k-best scores
        :return: np.array of size (n, t, k). see build_tree_2 for more details.
        """
        tree = self.build_tree_2(k_best)

        for i in range(len(self.sentence) + 1):
            if i == 0:
                idx = 0
                for j in self.t:
                    if ("START", j) not in self.transition or (j, self.sentence[0]) not in self.emission:
                        tree[idx, i, 0] = 0
                    else:
                        tree[idx, i, 0] = self.transition[("START", j)] * self.emission[(j, self.sentence[0])]
                    idx += 1

            elif i == len(self.sentence):
                all_scores = []
                idx = 0
                for j in self.t:
                    if (j, "STOP") not in self.transition:
                        all_scores.append([0])
                    else:
                        score = tree[idx, -1] * self.transition[(j, "STOP")]
                        all_scores.append(score)
                    idx += 1

                final_score = []
                flat_scores = list(itertools.chain(*all_scores))
                for l in range(k_best):
                    maxi = max(flat_scores)
                    final_score.append(maxi)
                    flat_scores.remove(maxi)


            else:
                idx = 0
                for j in self.t:
                    all_scores = []
                    idx_k = 0
                    for k in self.t:
                        print("Inner: " + k)
                        if (k, j) not in self.transition or (j, self.sentence[i]) not in self.emission:
                            all_scores.append([0])
                        else:
                            score = tree[idx_k, i-1] * self.transition[(k, j)] * self.emission[(j, self.sentence[i])]
                            all_scores.append(score)
                        idx_k += 1
                        # tree[idx, i, ] = max(all_scores)

                    flat_scores = list(itertools.chain(*all_scores))
                    for l in range(k_best):
                        maxi = max(flat_scores)
                        tree[idx, i, l] = maxi
                        flat_scores.remove(maxi)
                    idx += 1
        return tree, final_score





# sentence = "Municipal bonds are generally much safer than corporate bonds"
# df_en = read_to_pdf("EN/train")
# df_en = smoothingtrain(df_en)




def get_transition_lookup(df_transition):
    test = df_transition.drop(columns=["count_tag", "count_transition"])
    transition_lookup = dict()
    for i in range(len(test.tags)):
        transition_lookup[(test.tags[i], test.tags_next[i])] = test.transition_probability[i]
    return transition_lookup

def get_emission_lookup(df_emission):
    emission_lookup = dict()
    for i in range(len(df_emission.tags)):
        emission_lookup[(df_emission.tags[i], df_emission.words[i])] = df_emission.emission[i]
    return emission_lookup



# df_emission = estimate_emission_parameters(df_en)
# emission_lookup = get_emission_lookup(df_emission)
#
# df_transition = estimate_transition_parameters(df_en)
# transmission_lookup = get_transition_lookup(df_transition)
# print(emission_lookup)
# tags = list(set(df_transition.tags))
#
# test = viterbi(emission_lookup, transmission_lookup, sentence, tags)
# print(test.get_score(9, 1))


transition_lookup = {("START", "A"): 1.0, ("A", "A"): 0.5 , ("A", "B"): 0.5, ("B", "B"): 0.8, ("B", "STOP"): 0.2}
emission_lookup = {("A", "the"): 0.9, ("A", "dog"): 0.1, ("B", "the"): 0.1, ("B", "dog"): 0.9}

sentence = "the dog the"
print(sentence.split(" "))
test = viterbi(emission_lookup, transition_lookup, sentence, ["A", "B"])
print(test.populate_tree())
print(test.populate_tree_2(1))
# print(test.populate_tree_2(2))
# print(test.path)