from collections import Counter

from gensim.models import Word2Vec
import pandas as pd

if __name__ == '__main__':
    train_df = pd.read_csv('../../input/train_set.csv')
    test_df = pd.read_csv('../../input/test_set.csv')

    train_words = train_df['word_seg'].values.tolist()
    test_words = test_df['word_seg'].values.tolist()
    all_words = train_words + test_words

    train_chars = train_df['article'].values.tolist()
    test_chars = test_df['article'].values.tolist()
    all_chars = train_chars + test_chars

    with open('../../data/all_words.txt', 'w') as f:
        for text in all_words:
            f.write(f'{text}\n')

    with open('../../data/all_chars.txt', 'w') as f:
        for text in all_chars:
            f.write(f'{text}\n')