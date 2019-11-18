import pandas as pd

en_path = 'EN/train'

# def clean_data(data):

def read_to_pdf(file_path):
    with open(file_path) as f_message:
        temp = f_message.read().splitlines()
        temp = list(filter(None, temp))
    separated_word_tags = [word_tags.split(' ') for word_tags in temp]
    df = pd.DataFrame(separated_word_tags, columns =['words', 'tags'])
    return df


def replaceword(word, word_counts, k):
    if word_counts[word] < k:
        return "#UNK#"
    return word

def smoothing(data, k=3):
    word_counts = data['words'].value_counts().to_dict()
    data['words'] = data['words'].apply(lambda word: replaceword(word, word_counts, k))
    return data


def estimate_emission_parameters(word, tag, df):
    count_x_given_y = df[df['tags']==tag].words.str.count(word).sum()
    count_y = df.tags.str.count(tag).sum()
    return count_x_given_y/count_y


if __name__=="__main__":
    '''Part 2 Qn 1: Test MLE'''
    df = read_to_pdf(en_path)
    df = smoothing(df)
    print(estimate_emission_parameters('stress-related', 'B-NP', df))
