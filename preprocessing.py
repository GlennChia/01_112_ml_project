import pandas as pd
import copy
import numpy as np


class clean_trainset():
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = self.readtopdftrain()
        self.smoothed = self.smoothingtrain()
        # self.emission_df = self.estimate_emission_parameters()
        # self.emission_lookup = self.get_emissionlookup()
        # self.transition_df = self.estimate_transition_parameters()
        # self.transition_lookup =


    def readtopdftrain(self):
        with open(self.file_path, encoding="utf8") as f_message:
            temp = f_message.read().splitlines()
            temp = list(filter(None, temp))
        separated_word_tags = [word_tags.split(' ') for word_tags in temp]
        # separated_word_tags = [word.strip() for l in separated_word_tags for word in l]
        df = pd.DataFrame(separated_word_tags, columns=['words', 'tags'])
        df["words"] = [i.strip() for i in df.words]
        return df

    def replaceword(self, word, word_counts, k):
        if word_counts[word] < k:
            return "#UNK#"
        return word

    def smoothingtrain(self, k=3):
        word_counts = self.raw['words'].value_counts().to_dict()
        self.raw['words'] = self.raw['words'].apply(lambda word: self.replaceword(word, word_counts, k))
        return self.raw


    def estimate_emission_parameters(self):
        """
        Calculates the emission probabilities from count of words/ count of tags
        :param df: raw word to tag map
        :return: columns = count of word, count of tags, all emission probabilites of tag --> word
        """
        count_emit = self.smoothed.groupby(['tags', 'words']).size().reset_index()
        count_emit.columns = ["tags", "words", "count_emit"]
        count_tags = self.smoothed.groupby(["tags"]).size().reset_index()
        count_tags.columns = ["tags", "count_tags"]

        count = pd.merge(count_emit, count_tags, on="tags")
        count["emission"] = count["count_emit"] / count["count_tags"]
        return count.drop(columns=["count_emit", "count_tags"])

    def get_emissionlookup(self):
        """
        Map each word to tag of highest emission probability. This ensures lookup is in O(1) time.
        :param argmax_emission: Dataframe with emission probabilities of each tag --> word
        :return: Dictionary of word --> highest e(x|y) tag
        """
        idx = self.emission_df.groupby(['words'])['emission'].transform(max) == self.emission_df['emission']
        argmax_emission = self.emission_df[idx]
        lookup = dict(zip(argmax_emission.words, argmax_emission.tags))
        return lookup

    def estimate_transition_parameters(self):
        """Return a dataframe with
        tag | next_tag | count_tag | count_transition | transition_prob

        Parameters:
        df (DataFrame): Dataframe with word, tags, tags_next

        Returns:
        df (DataFrame)
        """
        out = copy.copy(self.smoothed)
        out['count_tag'] = out.groupby(['tags']).tags.transform(np.size)
        out['count_transition'] = out.groupby(['tags', 'tags_next']).tags.transform(np.size)
        out.loc[out.tags_next == '', 'count_transition'] = 0
        out['transition_probability'] = out['count_transition'] / out['count_tag']
        out = out.drop_duplicates(subset=['tags', 'tags_next'])
        out = out.drop(['words'], axis=1)
        out = out.sort_values(['tags', 'tags_next'])
        out = out.reset_index()
        out = out.drop(['index'], axis=1)
        return out


cleandata = clean_trainset("EN/train")
print(cleandata.smoothed)