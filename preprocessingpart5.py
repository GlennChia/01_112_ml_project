import pandas as pd
import copy
import numpy as np
import re

class clean_trainset():
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = self.read_to_pdf()
        self.smoothed = self.smoothingtrain()

        self.tags = self.get_tags()


    def read_to_pdf(self):
        with open(self.file_path, encoding='utf8') as f_message:
            temp = f_message.read().splitlines()
        words = []
        tags = []
        sentenceidlist = []
        sentenceid = 0
        for index, word_tags in enumerate(temp):
            if index == 0:
                words.append('')
                tags.append('START')
                sentenceidlist.append(sentenceid)

                split_word_tags = word_tags.split(' ')
                words.append(split_word_tags[0])
                tags.append(split_word_tags[1])
                sentenceidlist.append(sentenceid)

            elif word_tags == '':
                words.append('')
                tags.append('START')
                sentenceid += 1
                sentenceidlist.append(sentenceid)

            else:
                split_word_tags = word_tags.split(' ')
                words.append(split_word_tags[0])
                tags.append(split_word_tags[1])
                sentenceidlist.append(sentenceid)
        tags_next = tags[1:]
        df = pd.DataFrame(list(zip(words, tags, tags_next, sentenceidlist)), columns=['words', 'tags', 'tags_next',
                                                                                      'sentence id'])
        df['tags_next'] = df['tags_next'].str.replace('START', '')
        return df

    def replaceword(self, word, word_counts, k):
        if word_counts[word] < k:
            return "#UNK#"
        # re_punc = r'^[^a-zA-Z0-9]+$'
        # re_hash = r'^#'
        # re_at = r'^@'
        # re_num = r'\d'
        # re_url = r'(^http:|\.com$)'
        #
        # puncpattern = re.compile(re_punc)
        # hashpattern = re.compile(re_hash)
        # atpattern = re.compile(re_at)
        # numpattern = re.compile(re_num)
        # urlpattern = re.compile(re_url)
        # if puncpattern.match(word):
        #     return 'PUNC'
        # elif hashpattern.match(word):
        #     return 'HASH'
        # elif atpattern.match(word):
        #     return 'AT'
        # elif numpattern.match(word):
        #     return 'NUM'
        # elif urlpattern.match(word):
        #     return 'URL'
        if word == '':
            return word

        if '-' in word and word[0] != '-':
            return 'HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return 'YEAR'
        elif word[0].isdigit():
            return 'DIGITS'

        return word


    def smoothingtrain(self, k=2):
        word_counts = self.raw['words'].value_counts().to_dict()
        self.raw['words'] = self.raw['words'].apply(lambda word: self.replaceword(word, word_counts, k))
        return self.raw


    def outputsmootheddata(self):
        output = []
        sentence = []
        sentenceid = 0
        for row in self.smoothed.itertuples():
            if row[4] == sentenceid:
                if row[2] != 'START':
                    sentence.append((row[1], row[2]))
            else:
                output.append(sentence)
                sentence = []
                sentenceid += 1
        return output


    def get_tags(self):
        return list(set(self.smoothed["tags"]))

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
        re_punc = r'^[^a-zA-Z0-9]+$'
        re_hash = r'^#'
        re_at = r'^@'
        re_num = r'\d'
        re_url = r'(^http:|\.com$)'

        # puncpattern = re.compile(re_punc)
        # hashpattern = re.compile(re_hash)
        # atpattern = re.compile(re_at)
        # numpattern = re.compile(re_num)
        # urlpattern = re.compile(re_url)
        if word == '':
            return word

        if '-' in word and word[0] != '-':
            return 'HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return 'YEAR'
        elif word[0].isdigit():
            return 'DIGITS'
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

print(cleandata.outputsmootheddata()[0:4])
print(cleantest.get_all_sentences())