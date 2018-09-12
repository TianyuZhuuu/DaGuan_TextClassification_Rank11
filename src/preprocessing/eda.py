import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

if __name__ == '__main__':
    train_df = pd.read_csv('../../input/train_set.csv')
    test_df = pd.read_csv('../../input/test_set.csv')

    train_char = train_df['article'].values.tolist()
    train_word = train_df['word_seg'].values.tolist()
    train_label = train_df['class'].values
    test_char = test_df['article'].values.tolist()
    test_word = test_df['word_seg'].values.tolist()

    num_label = len(set(train_label))
    print(f'# of labels: {num_label}')

    train_char_len = [len(chars.split()) for chars in train_char]
    train_word_len = [len(words.split()) for words in train_word]
    print('Training set')
    print(np.percentile(train_word_len, [0, 50, 80, 90, 95, 98, 100]))
    print(np.percentile(train_char_len, [0, 50, 80, 90, 95, 98, 100]))

    test_char_len = [len(chars.split()) for chars in test_char]
    test_word_len = [len(words.split()) for words in test_word]
    print('Test set')
    print(np.percentile(test_word_len, [0, 50, 80, 90, 95, 98, 100]))
    print(np.percentile(test_char_len, [0, 50, 80, 90, 95, 98, 100]))
