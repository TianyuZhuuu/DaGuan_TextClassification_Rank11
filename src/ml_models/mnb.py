from sklearn.naive_bayes import MultinomialNB

from src.ml_models.utils import generate_oof_pred

if __name__ == '__main__':
    clf = MultinomialNB(alpha=0.0009)
    generate_oof_pred(clf, 'multinomialNB')
