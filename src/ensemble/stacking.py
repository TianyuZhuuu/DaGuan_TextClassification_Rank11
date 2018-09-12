import warnings

warnings.filterwarnings('ignore')
import os
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def get_layer1_input():
    fasttext_train = np.load('../../oof_pred/fasttext/1/fasttext_trainable_train_0.7588.npy')
    fasttext_test = np.load('../../oof_pred/fasttext/1/fasttext_trainable_test_0.7588.npy')

    fasttext_char_train1 = np.load('../../oof_pred/fasttext_char/1/fasttext_char_train_0.7156.npy')
    fasttext_char_test1 = np.load('../../oof_pred/fasttext_char/1/fasttext_char_test_0.7156.npy')

    fasttext_char_train = fasttext_char_train1
    fasttext_char_test = fasttext_char_test1

    # high_dropout_rnn_train = np.load(
    #     '../../oof_pred/high_dropout_pooled_bilstm/1/high_dropout_pooled_bilstm_train_0.7611.npy')
    # high_dropout_rnn_test = np.load(
    #     '../../oof_pred/high_dropout_pooled_bilstm/1/high_dropout_pooled_bilstm_test_0.7611.npy')

    # pooled_bilstm_train = np.load('../../oof_pred/pooled_bilstm/1/pooled_bilstm_train_0.7623.npy')
    # pooled_bilstm_test = np.load('../../oof_pred/pooled_bilstm/1/pooled_bilstm_test_0.7623.npy')
    #
    # pooled_bilstm_2layer_train = np.load('../../oof_pred/pooled_bilstm_2layer/1/pooled_bilstm_2layer_train_0.7647.npy')
    # pooled_bilstm_2layer_test = np.load('../../oof_pred/pooled_bilstm_2layer/1/pooled_bilstm_2layer_test_0.7647.npy')

    textcnn_train = np.load('../../oof_pred/textcnn/1/textcnn_train_0.7602.npy')
    textcnn_test = np.load('../../oof_pred/textcnn/1/textcnn_test_0.7602.npy')

    textcnn_char_train = np.load('../../oof_pred/textcnn_char/1/textcnn_char_train_0.7288.npy')
    textcnn_char_test = np.load('../../oof_pred/textcnn_char/1/textcnn_char_test_0.7288.npy')

    # textgru_ultimate_train = np.load('../../oof_pred/textgru_ultimate/1/textgru_ultimate_train_0.7665.npy')
    # textgru_ultimate_test = np.load('../../oof_pred/textgru_ultimate/1/textgru_ultimate_test_0.7665.npy')

    # textgrucnn_train = np.load('../../oof_pred/textgrucnn/1/textgrucnn_train_0.7683.npy')
    # textgrucnn_test = np.load('../../oof_pred/textgrucnn/1/textgrucnn_test_0.7683.npy')

    lgbm_svd_train = np.load('../../oof_pred/lgbm_svd_train_0.7373.npy')
    lgbm_svd_test = np.load('../../oof_pred/lgbm_svd_test_0.7373.npy')

    linearsvc_svd_train = np.load('../../oof_pred/linearsvc_svd_train_0.7078.npy')
    linearsvc_svd_test = np.load('../../oof_pred/linearsvc_svd_test_0.7078.npy')

    linearsvc_train = np.load('../../oof_pred/linearsvc_word_train_0.7772.npy')
    linearsvc_test = np.load('../../oof_pred/linearsvc_word_test_0.7772.npy')

    lr_train = np.load('../../oof_pred/lr_word_train_0.7720.npy')
    lr_test = np.load('../../oof_pred/lr_word_test_0.7720.npy')

    multinomialnb_train = np.load('../../oof_pred/multinomialNB_word_train_0.7375.npy')
    multinomialnb_test = np.load('../../oof_pred/multinomialNB_word_test_0.7375.npy')

    rnn_based_model_train = np.load('../../oof_pred/rnn_based_model_train_0.7817.npy')
    rnn_based_model_test = np.load('../../oof_pred/rnn_based_model_test_0.7817.npy')

    model_names = ['fasttext', 'fasttext_char', 'rnn_based_model', 'textcnn', 'textcnn_char',
                   'lgbm_svd', 'linearsvc_svd', 'linear_svc', 'lr', 'mnb']
    train_data = (
        fasttext_train, fasttext_char_train, rnn_based_model_train, textcnn_train, textcnn_char_train,
        lgbm_svd_train,
        linearsvc_svd_train, linearsvc_train, lr_train, multinomialnb_train)
    train_label = np.load('../../data/label.npy')
    test_x = (
        fasttext_test, fasttext_char_test, rnn_based_model_test, textcnn_test, textcnn_char_test, lgbm_svd_test,
        linearsvc_svd_test, linearsvc_test, lr_test, multinomialnb_test)

    train_data = np.hstack(train_data)
    test_x = np.hstack(test_x)

    return train_data, train_label, test_x


def stacking_layer1_oof_pred(model, model_name, train_data, train_label, test_x, num_fold, layer=1):
    fold_len = train_data.shape[0] // num_fold
    skf_indices = []
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=2018)
    for i, (train_idx, valid_idx) in enumerate(skf.split(np.ones(train_data.shape[0]), train_label)):
        skf_indices.extend(valid_idx.tolist())
    train_pred = np.zeros((train_data.shape[0], 19))
    test_pred = np.zeros((test_x.shape[0], 19))
    for fold in range(num_fold):
        print(f'Processing fold {fold}...')
        fold_start = fold * fold_len
        fold_end = (fold + 1) * fold_len
        if fold == num_fold - 1:
            fold_end = train_data.shape[0]
        train_indices = skf_indices[:fold_start] + skf_indices[fold_end:]
        test_indices = skf_indices[fold_start:fold_end]
        train_x = train_data[train_indices]
        train_y = train_label[train_indices]
        cv_test_x = train_data[test_indices]
        model.fit(train_x, train_y)

        pred = model.predict_proba(cv_test_x)
        train_pred[test_indices] = pred
        pred = model.predict_proba(test_x)
        test_pred += pred / num_fold

    y_pred = np.argmax(train_pred, axis=1)
    score = f1_score(train_label, y_pred, average='macro')
    print(score)

    pred_dir = f'../../oof_pred/layer{layer}_{model_name}/{model.random_state}/'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    train_path = pred_dir + f'train_{score:.6f}'
    test_path = pred_dir + f'test_{score:.6f}'

    np.save(train_path, train_pred)
    np.save(test_path, test_pred)


if __name__ == '__main__':
    train_data, train_label, test_data = get_layer1_input()

    # shallow lgbm
    for i in range(1, 11):
        model = LGBMClassifier(num_leaves=7, learning_rate=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                               random_state=i)
        stacking_layer1_oof_pred(model, f'lgbm_7leaves', train_data, train_label, test_data, 10)

    # medium lgbm
    for i in range(1, 11):
        model = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                               random_state=i)
        stacking_layer1_oof_pred(model, f'lgbm_31leaves', train_data, train_label, test_data, 10)

    # deep lgbm
    for i in range(1, 11):
        model = LGBMClassifier(num_leaves=127, learning_rate=0.05, n_estimators=500, subsample=0.8,
                               colsample_bytree=0.8, random_state=i)
        stacking_layer1_oof_pred(model, f'lgbm_127leaves', train_data, train_label, test_data, 10)
