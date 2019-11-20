import pandas as pd

en_path = 'EN/train'


def readtopdftrain(file_path):
    with open(file_path) as f_message:
        temp = f_message.read().splitlines()
        temp = list(filter(None, temp))
    separated_word_tags = [word_tags.split(' ') for word_tags in temp]
    df = pd.DataFrame(separated_word_tags, columns=['words', 'tags'])
    return df


def readtopdftest(file_path):
    with open(file_path) as f_message:
        temp = f_message.read().splitlines()
        temp = list(filter(None, temp))
    df = pd.DataFrame(temp, columns=['words'])
    return df


def replacewordtrain(word, word_counts, k):
    if word_counts[word] < k:
        return "#UNK#"
    return word


def smoothingtrain(data, k=3):
    word_counts = data['words'].value_counts().to_dict()
    data['words'] = data['words'].apply(lambda word: replacewordtrain(word, word_counts, k))
    return data


def smoothingtest(testdata, traindata):
    trainvalues = set(traindata['words'])
    testdata['words'] = testdata['words'].apply(lambda word: replacewordtest(word, trainvalues))
    return testdata


def replacewordtest(word, train):
    if word in train:
        return word
    return "#UNK#"


def estimate_emission_parameters(word, tag, df):
    count_x_given_y = df[df['tags'] == tag].words.str.count(word).sum()
    count_y = df.tags.str.count(tag).sum()
    return count_x_given_y/count_y


if __name__=="__main__":
    '''Part 2 Qn 1: Test MLE'''
    traindf = readtopdftrain(en_path)
    smoothedtrain = smoothingtrain(traindf)
    testfilepath = 'EN/dev.in'
    testdf = readtopdftest(testfilepath)
    smoothedtest = smoothingtest(testdf, smoothedtrain)
    print(estimate_emission_parameters('stress-related', 'B-NP', smoothedtrain))
    print(smoothedtest['words'].value_counts())
