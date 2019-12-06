import pandas as pd
import copy
import numpy as np


class clean_trainset():
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = self.read_to_pdf()
        self.smoothed = self.smoothingtrain()

        self.emission_df = self.estimate_emission_parameters()
        self.emission_lookup = self.get_emissionlookup()

        self.transition_df = self.estimate_transition_parameters()
        self.transition_lookup = self.get_transition_lookup()

        self.tags = self.get_tags()


    def read_to_pdf(self):
        with open(self.file_path, encoding='utf8') as f_message:
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
        df = pd.DataFrame(list(zip(words, tags, tags_next)), columns=['words', 'tags', 'tags_next'])
        df['tags_next'] = df['tags_next'].str.replace('START', '')
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
        return {(i, j ): k for i, j, k in
                zip(self.emission_df["tags"],
                    self.emission_df["words"],
                    self.emission_df["emission"])}

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

    def get_transition_lookup(self):
        return {(i, j): k for i, j, k in
                zip(self.transition_df["tags"],
                    self.transition_df["tags_next"],
                    self.transition_df["transition_probability"])}

    def get_tags(self):
        return list(set(self.transition_df["tags"]))

class clean_testset():
    def __init__(self, train_path, train_df):
        self.train_path = train_path
        self.raw, self.size = self.read_to_pdf()
        self.smoothed = self.smoothingtest(train_df)

    def read_to_pdf(self):
        with open(self.train_path, encoding="utf8") as f_message:
            temp = f_message.read().splitlines()

        words = []
        sentenceid = 0
        for word in temp:
            if word != "":
                words.append([word, sentenceid])
            else:
                sentenceid += 1

        df = pd.DataFrame(words, columns=['words', 'sentence_id'])

        return df, sentenceid


    def smoothingtest(self, train_df):
        trainvalues = set(train_df['words'])
        self.raw['words'] = self.raw['words'].apply(lambda word: self.replaceword(word, trainvalues))
        return self.raw

    def replaceword(self, word, train):
        if word in train:
            return word
        return "#UNK#"

    def get_all_sentences(self):
        all_sentences = []

        sentence = []
        s_id = 0
        for enum, row in self.smoothed.iterrows():
            if row.sentence_id == s_id:
                sentence.append(row.words)
            else:
                all_sentences.append(sentence)
                sentence = [row.words]
                s_id += 1
        all_sentences.append(sentence)
        return all_sentences



cleandata = clean_trainset("EN/train")
cleantest = clean_testset("EN/dev.in", cleandata.smoothed)

print(cleantest.smoothed)
print(cleantest.get_all_sentences()[-1])