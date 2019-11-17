import pandas as pd

en_path = 'EN/train'


def read_to_pdf(file_path):
    with open(file_path) as f_message:
        temp = f_message.read().splitlines()
        temp = list(filter(None, temp))
    separated_word_tags = [word_tags.split(' ') for word_tags in temp]
    df = pd.DataFrame(separated_word_tags, columns =['words', 'tags'])
    return df


if __name__=="__main__":
    '''Part 2 Qn 1: Test MLE'''
    df = read_to_pdf(en_path)
    print(df)
