import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 2018
np.random.seed(SEED)

if __name__ == '__main__':
    train_df = pd.read_csv('../../input/train_set.csv')
    test_df = pd.read_csv('../../input/test_set.csv')

    train_char = train_df['article'].values.tolist()
    train_word = train_df['word_seg'].values.tolist()
    train_label = train_df['class'].values - 1
    test_char = test_df['article'].values.tolist()
    test_word = test_df['word_seg'].values.tolist()

    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    train_word_feat = word_vectorizer.fit_transform(train_word)
    test_word_feat = word_vectorizer.transform(test_word)

    svd = TruncatedSVD(n_components=100, n_iter=20, random_state=SEED)
    train_svd_feat = svd.fit_transform(train_word_feat)
    print('Training set transformed..')
    with open('../../data/train_svd_feat.pkl', 'wb') as f:
        pkl.dump(train_svd_feat, f)

    test_svd_feat = svd.transform(test_word_feat)
    print('Test set transformed..')
    with open('../../data/test_svd_feat.pkl', 'wb') as f:
        pkl.dump(test_svd_feat, f)