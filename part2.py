import pandas as pd



def readtopdftrain(file_path):
    with open(file_path, encoding="utf8") as f_message:
        temp = f_message.read().splitlines()
        temp = list(filter(None, temp))
    separated_word_tags = [word_tags.split(' ') for word_tags in temp]
    # separated_word_tags = [word.strip() for l in separated_word_tags for word in l]
    df = pd.DataFrame(separated_word_tags, columns=['words', 'tags'])
    df["words"] = [i.strip() for i in df.words]
    return df

def readtopdftest(file_path):
    with open(file_path, encoding="utf8") as f_message:
        temp = f_message.read().splitlines()

    words = []
    sentenceid = 0
    for word in temp:
        if word != "":
            words.append([word, sentenceid])
        else:
            sentenceid += 1

    df = pd.DataFrame(words, columns=['words', 'sentence id'])
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


def get_emissionlookup(argmax_emission):
    """
    Map each word to tag of highest emission probability. This ensures lookup is in O(1) time.
    :param argmax_emission: Dataframe with emission probabilities of each tag --> word
    :return: Dictionary of word --> highest e(x|y) tag
    """
    ref_df = argmax_emission.groupby(["words"]).max(axis=["emission"]).reset_index()
    lookup = dict(zip(ref_df.words, ref_df.tags))
    return lookup


def get_tag_fromemission(lookup, smoothedtest, dataset):
    """
    Retrieve tag for each word seen in testset
    :param lookup: word --> Highest e(y|x) tag
    :param smoothedtest: Processed testset to exclude non-occuring words in trainset
    :param dataset: EN/ CN/ AL/ SG
    :return: output file with allocated tags
    """
    output_file = dataset + "/dev.p2.out"
    with open(output_file, "w", encoding="utf8") as f:
        sentenceid = 0
        for index, row in smoothedtest.iterrows():
            if row['sentence id'] != sentenceid:
                f.write("\n")
                f.write(row['words'] + " " + lookup[row['words']] + "\n")
                sentenceid += 1
            else:
                f.write(row['words'] + " " + lookup[row['words']] + "\n")
    f.close()

def sentiment_analysis(dataset):
    train_path = dataset + "/train"
    traindf = readtopdftrain(train_path)
    smoothedtrain = smoothingtrain(traindf)
    test_path = dataset + "/dev.in"
    testdf = readtopdftest(test_path)
    smoothedtest = smoothingtest(testdf, smoothedtrain)
    argmax_emission = estimate_emission_parameters(smoothedtrain)
    lookup = get_emissionlookup(argmax_emission)
    get_tag_fromemission(lookup, smoothedtest, dataset)

    print("Done with dataset " + train_path)

if __name__=="__main__":
    '''Part 2 Qn 1: Test MLE'''

    sentiment_analysis("EN")
    # sentiment_analysis("CN")
    # sentiment_analysis("AL")
    # sentiment_analysis("SG")


