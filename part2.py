import pandas as pd

en_path = 'EN/train'


def read_to_pdf(file_path):
    with open(file_path) as f_message:
        temp = f_message.read().splitlines()
        temp = list(filter(None, temp))
    separated_word_tags = [word_tags.split(' ') for word_tags in temp]
    df = pd.DataFrame(separated_word_tags, columns =['words', 'tags'])
    return df


def estimate_emission_parameters(word, tag, df):
    count_x_given_y = df[df['tags']==tag].words.str.count(word).sum()
    count_y = df.tags.str.count(tag).sum()
    return count_x_given_y/count_y


if __name__=="__main__":
    '''Part 2 Qn 1: Test MLE'''
    df = read_to_pdf(en_path)
    print(estimate_emission_parameters('stress-related', 'B-NP', df))