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

    MAX_LEN = 2000
    EMBED_DIM = 300
    EMBED_PATH = '../../data/word_vector_300d.vec'
    NUM_WORDS = 359279

    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(train_word + test_word)
    train_sequence = pad_sequences(tokenizer.texts_to_sequences(train_word), MAX_LEN)
    test_sequence = pad_sequences(tokenizer.texts_to_sequences(test_word), MAX_LEN)

    word2vec_mapping = {}
    with open(EMBED_PATH, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            vals = line.split()
            word2vec_mapping[vals[0]] = np.fromiter(vals[1:], dtype=np.float32)
    print(f'# of words: {len(word2vec_mapping)}')

    oov = 0
    embed_mat = np.random.uniform(-0.1, 0.1, (NUM_WORDS + 1, EMBED_DIM))
    for word, i in tokenizer.word_index.items():
        if i > NUM_WORDS:
            break
        else:
            if word in word2vec_mapping:
                embed_mat[i, :] = word2vec_mapping[word]
            else:
                print(i)
                oov += 1
    print(f'# of OOV words: {oov}')

    np.save('../../data/train_input', train_sequence)
    np.save('../../data/test_input', test_sequence)
    np.save('../../data/word_embed_mat', embed_mat)
