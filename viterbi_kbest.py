import numpy as np
import preprocessing


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
    def __init__(self, emission, transition, sentence, tags, kbest):
        """
        :param emission: lookup of emission probabilities (dict) (tag, word) --> probability
        :param transition: lookup of transmission probabilities (dict) (tag1, tag2) --> probability
        :param sentence: string sentence
        :param tags: array of tags available for dataset
        """
        self.emission = emission
        self.transition = transition
        self.sentence = sentence
        self.n = len(sentence) + 2
        self.t = tags
        self.kbest = kbest
        # self.tree_k, self.finalscore_k, self_kpaths = self.populate_tree_2(2)
        self.kth_path = self.populate_tree_2()

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
            if (j, dest_node) in self.transition:
                out[idx] = self.transition[j, dest_node]
            idx += 1
        return out



    def populate_tree_2(self):
        """
        Returns populated tree with k-best scores for each node
        :param k: choice of k-best scores
        :return: np.array of size (n, t, k). see build_tree_2 for more details.
        """
        tree = self.build_tree_2(self.kbest)
        all_nodes = [[[node_k(i, j,  k) for k in range(self.kbest)]for j in range(len(self.t))] for i in range(len(self.sentence))]

        for i in range(len(self.sentence) + 1):
            if i == 0:
                idx = 0
                for j in self.t:
                    if ("START", j) not in self.transition or (j, self.sentence[0]) not in self.emission:
                        tree[i, idx, 0] = 0
                    else:
                        tree[i, idx, 0] = self.transition[("START", j)] * self.emission[(j, self.sentence[0])]
                    idx += 1


            elif i == len(self.sentence):
                last_layer = []

                raw_scores = tree[i-1]  # extracts column (k x 1) scores in i-1
                candidate_scores = raw_scores * self.get_transition_array("STOP")
                max_ind = k_largest_index_argsort(candidate_scores, self.kbest)
                final_score = [candidate_scores[tuple(idx)] for idx in max_ind]


                for l in range(self.kbest):
                    last_node = node_k(i, 0, 0)
                    j_coord = max_ind[l][0]
                    k_coord = max_ind[l][1]

                    last_node.parent = all_nodes[i - 1][j_coord][k_coord]
                    last_node.subpath = self.t[j_coord]
                    last_layer.append(last_node)

            else:
                raw_scores = tree[i-1]  # extracts column (k x 1) scores in i-1

                idx_k = 0
                for k in self.t:
                    candidate_scores = raw_scores * self.get_transition_array(k)
                    max_ind = k_largest_index_argsort(candidate_scores, self.kbest)

                    if (k, self.sentence[i]) not in self.emission:
                        tree[i, idx_k] = 0
                    else:
                        tree[i, idx_k] = np.array([candidate_scores[tuple(idx)] for idx in max_ind]) * self.emission[k, self.sentence[i]]

                    for l in range(self.kbest):
                        j_coord = max_ind[l][0]
                        k_coord = max_ind[l][1]

                        current_node = all_nodes[i][idx_k][l]
                        current_node.parent = all_nodes[i-1][j_coord][k_coord]
                        current_node.subpath = self.t[j_coord]
                    idx_k += 1

        # ---! following commented lines return ALL k best paths !---
        # k_paths = []
        # for a in range(k_best):
        #     path = []
        #     current_node = last_layer[a]
        #     for i in range(len(self.sentence), 0, -1):
        #         path.insert(0, current_node.subpath)
        #         current_node = current_node.parent
        #
        #     k_paths.append(path)

        kth_path = []
        kth_best = last_layer[-1]
        for i in range(len(self.sentence), 0, -1):
            kth_path.insert(0, kth_best.subpath)
            kth_best = kth_best.parent

        # return tree, final_score, k_paths
        return kth_path

transition_lookup = {("START", "A"): 1.0, ("A", "A"): 0.5 , ("A", "B"): 0.5, ("B", "B"): 0.8, ("B", "STOP"): 0.2}
emission_lookup = {("A", "the"): 0.9, ("A", "dog"): 0.1, ("B", "the"): 0.1, ("B", "dog"): 0.9}
sentence = ["the", "dog", "the"]
test = viterbi(emission_lookup, transition_lookup, sentence, ["A", "B"], 1)
test2 = viterbi(emission_lookup, transition_lookup, sentence, ["A", "B"], 2)

print("1st best : ", test.populate_tree_2())
print("2nd best: ", test2.populate_tree_2())



def run_test(dataset):
    # cleandata = preprocessing.clean_trainset(dataset + "/train")
    # cleantest = preprocessing.clean_testset(dataset + "/dev.in", cleandata.smoothed)

    cleandata = preprocessing.clean_trainset(dataset + "/train")
    cleantest = preprocessing.clean_testset("Test/" + dataset + "/test.in", cleandata.smoothed)


    emission = cleandata.emission_lookup
    transition = cleandata.transition_lookup
    for sentence in cleantest.get_all_sentences():
        obj = viterbi(emission, transition, sentence, cleandata.tags, 1)
        pred_tags = obj.populate_tree_2()
        with open("Test/" + dataset + "/test.out", "a", encoding="utf8") as f:
            count = 0
            for word in sentence:
                f.write(word + " " + pred_tags[count] + "\n")
                count += 1
            f.write("\n")
#
# for d in ["EN", "CN", "AL", "SG"]:
#     run_test(d)

run_test("AL")