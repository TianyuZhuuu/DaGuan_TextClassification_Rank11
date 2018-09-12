import numpy as np
import pandas as pd
import pickle as pkl
import os

from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_word_data():
    if os.path.exists('../../data/train_word.pkl'):
        with open('../../data/train_word.pkl', 'rb') as f:
            train_x = pkl.load(f)
        with open('../../data/label.pkl', 'rb') as f:
            train_y = pkl.load(f)
        with open('../../data/test_word.pkl', 'rb') as f:
            test_x = pkl.load(f)
    else:
        train_df = pd.read_csv('../../input/train_set.csv')
        test_df = pd.read_csv('../../input/test_set.csv')
        train_word = train_df['word_seg'].values
        train_y = train_df['class'].values.astype(np.int) - 1
        test_word = test_df['word_seg'].values
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=3, sublinear_tf=True)
        train_x = vectorizer.fit_transform(train_word)
        test_x = vectorizer.transform(test_word)
        with open('../../data/train_word.pkl', 'wb') as f:
            pkl.dump(train_x, f)
        print('Training word data saved...')
        with open('../../data/label.pkl', 'wb') as f:
            pkl.dump(train_y, f)
        print('Label saved...')
        with open('../../data/test_word.pkl', 'wb') as f:
            pkl.dump(test_x, f)
        print('Test word data saved...')
    return train_x, train_y, test_x


def prepare_char_data():
    if os.path.exists('../../data/train_char.pkl'):
        with open('../../data/train_char.pkl', 'rb') as f:
            train_x = pkl.load(f)
        with open('../../data/label.pkl', 'rb') as f:
            train_y = pkl.load(f)
        with open('../../data/test_char.pkl', 'rb') as f:
            test_x = pkl.load(f)
    else:
        train_df = pd.read_csv('../../input/train_set.csv')
        test_df = pd.read_csv('../../input/test_set.csv')
        train_char = train_df['article'].values
        train_y = train_df['class'].values.astype(np.int) - 1
        test_char = test_df['article'].values
        vectorizer = TfidfVectorizer(ngram_range=(2, 4), max_df=0.9, min_df=3, sublinear_tf=True)
        train_x = vectorizer.fit_transform(train_char)
        test_x = vectorizer.transform(test_char)
        with open('../../data/train_char.pkl', 'wb') as f:
            pkl.dump(train_x, f)
        print('Training char data saved...')
        with open('../../data/label.pkl', 'wb') as f:
            pkl.dump(train_y, f)
        print('Label saved...')
        with open('../../data/test_char.pkl', 'wb') as f:
            pkl.dump(test_x, f)
        print('Test char data saved...')
    return train_x, train_y, test_x


def make_submission(pred, name):
    ids = pd.read_csv('../../submission/baseline.csv')['id']
    assert isinstance(pred, np.ndarray)
    if len(pred.shape) == 2:
        y_pred = np.argmax(pred, axis=1)
    else:
        y_pred = ids
    y_pred += 1
    submit = pd.DataFrame({'id': ids, 'class': y_pred})
    submit.to_csv(f'../../submission/{name}.csv', index=False)

    # if np.size()


if __name__ == '__main__':

    # Rank 1 : 0.798644
    # pred = np.load('../../oof_pred/all_depth_merged_lgbm_test_0.795714.npy')
    # make_submission(pred, 'all_depth_merged_lgbm_0.7957')

    # Rank 2 : 0.798309
    # pred = np.load('../../oof_pred/blend_v2_test_0.795332.npy')
    # make_submission(pred, 'Blend_v2_0.795332')

    # Rank 3 : 0.79***2
    # pred = np.load('../../oof_pred/robust_50lgbms_gmean_test_0.794848.npy')
    # make_submission(pred, 'robust_50lgbms_gmean_0.794848')

    # Rank 4: 0.798217
    # pred = np.load('../../oof_pred/Ultimate_blend_wo_newlgbm_test_0.795955.npy')
    # make_submission(pred, 'Ultimate_blend_wo_newlgbm_0.795955')

    # Rank 5 : 0.798065
    # pred = np.load('../../oof_pred/blend_v1_0.795363.npy')
    # make_submission(pred, 'Blend_v1_0.796363')

    # Rank 6 : 0.797812
    # pred = np.load('../../oof_pred/test_0.7941.npy')
    # make_submission(pred, 'deep_lgbm_10fold_no_validation_0.7941')

    # Rank 7 : 0.79***1
    # pred = np.load('../../oof_pred/robust_50lgbms_ensemble_test_0.795897.npy')
    # make_submission(pred, 'robust_50lgbm_0.795897')

    # Rank 8 : 0.797726
    # pred = np.load('../../oof_pred/Medium_lgbm_mean_test_0.794165.npy')
    # make_submission(pred, 'medium_lgbm_10fold_0.794165')

    # Rank 9 : 0.797662
    # pred = np.load('../../oof_pred/layer1_Deep_lgbm_strong_regularization/1/test_0.7940.npy')
    # make_submission(pred, 'deep_lgbm_20fold_validation_0.7940')

    # Rank 10 : 0.796457
    # pred = np.load('../../oof_pred/layer1_LGBM_10fold_test_0.795097.npy')
    # make_submission(pred, 'LGBM_10fold_0.795097')

    # Rank 11 : 0.79626
    # pred = np.load('../../oof_pred/layer1_LGBM_10fold/1/test_0.795009.npy')
    # make_submission(pred, 'LGBM_10fold_0.795009')

    # pred = np.load('../../oof_pred/layer1_best_lgbm_minchild50_test_0.794752.npy')
    # make_submission(pred, '32_best_lgbm_minchild50_0.4752')

    # pred = np.load('../../oof_pred/layer1_blend_test_0.791541.npy')
    # make_submission(pred, 'layer1_blend_0.791541')

    # 0.796319
    # pred = np.load('../../oof_pred/layer1_LGBM_20fold_test_0.795837.npy')
    # make_submission(pred, 'new_lgbm_mean_0.795837')

    # 0.796403
    # pred = np.load('../../oof_pred/9_New_lgbm_20fold_test_0.796588.npy')
    # make_submission(pred, '9_New_lgbm_20fold_0.796588')

    # pred = np.load('../../oof_pred/layer1_robust_lgbm_demo/1/test_0.789606.npy')
    # make_submission(pred, 'robust_lgbm_0.789606')

    # 0.795965
    # pred = np.load('../../oof_pred/new_stacking_test_0.7919.npy')
    # make_submission(pred, 'stacking_0.7919')

    # rank 1 0.79519 CV: 0.7891 LB-CV: 0.00609
    # pred = np.load('../../oof_pred/stacking_test_0.7891.npy')
    # make_submission(pred, 'stacking_0.7891')

    # rank 1 0.786274 CV: 0.7683 LB-CV: 0.017974
    # pred = np.load('../../oof_pred/textgrucnn/1/textgrucnn_test_0.7683.npy')
    # make_submission(pred, '10fold_textgrucnn_0.7683')

    # rank 2 0.784384 CV: 0.7665 LB-CV: 0.017884
    # pred = np.load('../../oof_pred/textgru_ultimate/1/textgru_ultimate_test_0.7665.npy')
    # make_submission(pred, '10fold_textgru_ultimate_0.7665')

    # rank 3 0.783885 CV: 0.7647 LB-CV: 0.019185
    # pred = np.load('../../oof_pred/pooled_bilstm_2layer/1/pooled_bilstm_2layer_test_0.7647.npy')
    # make_submission(pred, '10fold_pooled_bilstm_2layer_0.7647')

    # rank 4 0.779099 CV: 0.7623 LB-CV: 0.016799
    # pred = np.load('../../oof_pred/pooled_bilstm/1/pooled_bilstm_test_0.7623.npy')
    # make_submission(pred, '10fold_pooled_bilstm_0.7623')

    # rank 5 0.778444 CV: 0.7611 LB-CV:  0.017344
    # pred = np.load('../../oof_pred/high_dropout_pooled_bilstm/1/high_dropout_pooled_bilstm_test_0.7611.npy')
    # make_submission(pred, '10fold_high_dropout_bilstm_0.7611')

    # rank 6 0.777711 CV: 0.7772 LB-CV: 0.000511
    # pred = np.load('../../oof_pred/linearsvc_word_test_0.7772.npy')
    # make_submission(pred, '10fold_linearsvc_0.7772')

    # rank 7 0.77399 CV: 0.76020 LB-CV: 0.01379
    # pred = np.load('../../oof_pred/textcnn/1/textcnn_test_0.7602.npy')
    # make_submission(pred, '10fold_textcnn_0.7602')

    # rank 8 0.77**1 CV: 0.7720
    # pred = np.load('../../oof_pred/lr_word_test_0.7720.npy')
    # make_submission(pred, '10fold_lr_0.7720')

    # rank 9 0.766523 CV: 0.758800 LB-CV: 0.007723
    # pred =np.load('../../oof_pred/fasttext/1/fasttext_trainable_test_0.7588.npy')
    # make_submission(pred, '10fold_fasttext_2fc_0.7588')

    pass
