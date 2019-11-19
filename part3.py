import pandas as pd

en_path = 'EN/train copy'


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
    df = pd.DataFrame(list(zip(words, tags)), columns =['words', 'tags']) 
    return df


def estimate_transmission_parameters(word, tag, df):
    count_x_given_y = df[df['tags']==tag].words.str.count(word).sum()
    count_y = df.tags.str.count(tag).sum()
    return count_x_given_y/count_y


if __name__=="__main__":
    '''Part 3 Qn 1: Test transition parameters'''
    df = read_to_pdf(en_path)
    print(df)
    # df = smoothing(df)
    # print(estimate_emission_parameters('stress-related', 'B-NP', df))
