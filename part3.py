import pandas as pd
import numpy as np

en_path = 'EN/train'


def read_to_pdf(file_path):
    with open(file_path) as f_message:
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
    return df


if __name__=="__main__":
    '''Part 3 Qn 1: Test transition parameters'''
    df = read_to_pdf(en_path)
    print(df)
    print(estimate_transition_parameters(df))
