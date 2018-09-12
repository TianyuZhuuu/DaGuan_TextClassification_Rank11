import keras
import pandas as pd
import numpy as np

np.random.seed(2018)

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

if __name__ == '__main__':
    train_df = pd.read_csv('../../input/train_set.csv')
    test_df = pd.read_csv('../../input/test_set.csv')

    train_char = train_df['article'].values.tolist()
    train_word = train_df['word_seg'].values.tolist()
    train_label = train_df['class'].values - 1
    test_char = test_df['article'].values.tolist()
    test_word = test_df['word_seg'].values.tolist()

    np.save('../../data/label', train_label)

    # np.percentile(train_word_len, [0, 50, 80, 90, 95, 98, 100])
    # of labels: 19
    # Training set
    # [6.     514.     990.    1428.    1949.    2858.48 39759.]
    # [50.     842.    1618.    2346.    3201.    4720.96 55804.]
    # Test set
    # [6.   516.   992.  1429.  1949.  2826. 19755.]
    # [50.   842.  1621.  2349.  3207.  4672. 31694.]

    MAX_LEN = 3200
    EMBED_DIM = 300
    EMBED_PATH = '../../data/char_vector_300d_new.vec'

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_char + test_char)
    NUM_WORDS = len([w for w, c in tokenizer.word_counts.items() if c >= 5])
    print(NUM_WORDS)
    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(train_char + test_char)

    train_sequence = pad_sequences(tokenizer.texts_to_sequences(train_char), MAX_LEN)
    test_sequence = pad_sequences(tokenizer.texts_to_sequences(test_char), MAX_LEN)

    char2vec_mapping = {}
    with open(EMBED_PATH, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            vals = line.split()
            char2vec_mapping[vals[0]] = np.fromiter(vals[1:], dtype=np.float32)
    print(f'# of chars: {len(char2vec_mapping)}')

    oov = 0
    embed_mat = np.random.uniform(-0.1, 0.1, (len(char2vec_mapping) + 1, EMBED_DIM)).astype(np.float32)
    for char, i in tokenizer.word_index.items():
        if i > NUM_WORDS:
            break
        else:
            if char in char2vec_mapping:
                embed_mat[i, :] = char2vec_mapping[char]
            else:
                oov += 1
    print(f'# of OOV words: {oov}')

    np.save('../../data/char_train_input', train_sequence)
    np.save('../../data/char_test_input', test_sequence)
    np.save('../../data/char_embed_mat', embed_mat)
