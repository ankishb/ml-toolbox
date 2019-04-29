def run_xgb_target_enc(params, X_train, X_test):
    n_splits = 8
    verbose_eval = 1000
    num_rounds = 60000
    early_stop = 500

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    qwk_scores = []
    i = 0

    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):
        X_tst = X_test
        print("fold ",i, X_tst.shape, X_train.shape)
        X_tr, X_val = X_train.iloc[train_idx, :], X_train.iloc[valid_idx, :]
        y_tr, y_val = X_tr['AdoptionSpeed'], X_val['AdoptionSpeed']
        X_tr, X_val = X_tr.drop(['AdoptionSpeed'], axis=1), X_val.drop(['AdoptionSpeed'], axis=1)

        cols_ = ['Age', 'Breed1', 'Breed2','Quantity', 'PhotoAmt', 'sentiment_magnitude','sentiment_score',
                 'main_breed_BreedName', 'second_breed_BreedName','RescuerID_COUNT']

        tr_es, vl_es, ts_es = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for f in cols_:
            tr_, vl_, ts_ = target_encode(  trn_series=X_tr[f],val_series=X_val[f],tst_series=X_tst[f],
                                            target=y_tr,min_samples_leaf=70,
                                            smoothing=10, noise_level=0.01)
#             print(tr_.shape, vl_.shape, ts_.shape)
            tr_es = pd.concat([tr_es, tr_], axis=1)
            vl_es = pd.concat([vl_es, vl_], axis=1)
            ts_es = pd.concat([ts_es, ts_], axis=1)
#             print(tr_.shape, vl_.shape, ts_.shape)

        tr_es = pd.DataFrame(data=tr_es.values, columns=[col+'_en' for col in cols_])
        ts_es = pd.DataFrame(data=ts_es.values, columns=[col+'_en' for col in cols_])
        vl_es = pd.DataFrame(data=vl_es.values, columns=[col+'_en' for col in cols_])
#         print("check encoder data shape: ",tr_es.shape, ts_es.shape, vl_es.shape)
#         print("check test shape: ", X_test.shape, X_tst.shape)
#         print("training data shape: ", pd.concat([X_tr.reset_index(drop=True), tr_es.reset_index(drop=True)], axis=1).shape, y_tr.shape)
#         print("indices: ", X_tr.index, tr_es.index, y_tr.index)

        y_tr, y_val = y_tr.values, y_val.values
        y_tr_ = 0.15*np.random.randn(y_tr.shape[0]) + y_tr
        d_train = xgb.DMatrix(data=pd.concat([X_tr.reset_index(drop=True), tr_es.reset_index(drop=True)], axis=1), 
                              label=y_tr_, 
                              feature_names=pd.concat([X_tr.reset_index(drop=True), tr_es.reset_index(drop=True)], axis=1).columns)
        d_valid = xgb.DMatrix(data=pd.concat([X_val.reset_index(drop=True), vl_es.reset_index(drop=True)], axis=1), 
                              label=y_val, 
                              feature_names=pd.concat([X_val.reset_index(drop=True), vl_es.reset_index(drop=True)], axis=1).columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        valid_pred = model.predict(xgb.DMatrix(pd.concat([X_val.reset_index(drop=True), vl_es.reset_index(drop=True)], axis=1), 
                                               feature_names=pd.concat([X_val.reset_index(drop=True), vl_es.reset_index(drop=True)], axis=1).columns), 
                                   ntree_limit=model.best_ntree_limit)
        ############################
        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_val_y_k = optR.predict(valid_pred, coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_val_y_k)
        qwk_scores.append(qwk)
        print("QWK = ", qwk)
        #########################
#         print(pd.concat([X_tst.reset_index(drop=True), ts_es.reset_index(drop=True)], axis=1).shape, pd.concat([X_tst.reset_index(drop=True), ts_es.reset_index(drop=True)], axis=1))
        test_pred = model.predict(xgb.DMatrix(pd.concat([X_tst.reset_index(drop=True), ts_es.reset_index(drop=True)], axis=1), 
                                              feature_names=pd.concat([X_tst.reset_index(drop=True), ts_es.reset_index(drop=True)], axis=1).columns), 
                                  ntree_limit=model.best_ntree_limit)
        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred
        i += 1
    print(X_tst.shape, X_test.shape, X_train.shape)
    print("Avg qwk: ", np.mean(qwk_scores))
    return model, oof_train, oof_test









def run_lgb_target_enc(params, X_train, X_test):
    # Additional parameters:
    early_stop = 500
    verbose_eval = 400
    num_rounds = 15000
    n_splits = 8
    from sklearn.model_selection import StratifiedKFold, GroupKFold
#     kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)
    kfold = GroupKFold(n_splits=n_splits)#, random_state=1337)
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    qwk_scores = []
    i = 0
    for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values, X_train.RescuerID.values):
#     for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values):
        print("fold ",i, train_index.shape, valid_index.shape)
        X_tst = X_test
        X_tr = X_train.iloc[train_index, :]
        X_val = X_train.iloc[valid_index, :]

        y_tr = X_tr['AdoptionSpeed']#.values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)
        
        cols_ = ['Age', 'Breed1', 'Breed2','Quantity', 'PhotoAmt', 'sentiment_magnitude','sentiment_score',
                 'main_breed_BreedName', 'second_breed_BreedName','RescuerID_COUNT']

        tr_es, vl_es, ts_es = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for f in cols_:
            tr_, vl_, ts_ = target_encode(  trn_series=X_tr[f],
                                            val_series=X_val[f],
                                            tst_series=X_tst[f],
                                            target=y_tr,
                                            min_samples_leaf=50,
                                            smoothing=5,
                                            noise_level=0.01
                                            )
#             print(tr_.shape, vl_.shape, ts_.shape)
            tr_es = pd.concat([tr_es, tr_], axis=1)
            vl_es = pd.concat([vl_es, vl_], axis=1)
            ts_es = pd.concat([ts_es, ts_], axis=1)
#             print(tr_.shape, vl_.shape, ts_.shape)

        tr_es = pd.DataFrame(data=tr_es.values, columns=[col+'_en' for col in cols_])
        ts_es = pd.DataFrame(data=ts_es.values, columns=[col+'_en' for col in cols_])
        vl_es = pd.DataFrame(data=vl_es.values, columns=[col+'_en' for col in cols_])
#         print("check encoder data shape: ",tr_es.shape, ts_es.shape, vl_es.shape)
#         print("check test shape: ", X_test.shape, X_tst.shape)
#         print("training data shape: ", pd.concat([X_tr.reset_index(drop=True), tr_es.reset_index(drop=True)], axis=1).shape, y_tr.shape)
#         print("indices: ", X_tr.index, tr_es.index, y_tr.index)
        y_tr = y_tr.values  
        d_train = lgb.Dataset(pd.concat([X_tr.reset_index(drop=True), tr_es.reset_index(drop=True)], axis=1), label=y_tr)#, categorical_feature=cat_feat)
        d_valid = lgb.Dataset(pd.concat([X_val.reset_index(drop=True), vl_es.reset_index(drop=True)], axis=1), label=y_val)#, categorical_feature=cat_feat)
        watchlist = [d_train, d_valid]
        model = lgb.train(params, train_set=d_train, valid_sets=watchlist,
                          verbose_eval=verbose_eval, num_boost_round=num_rounds, 
                          early_stopping_rounds=early_stop)
        
        val_pred = model.predict(pd.concat([X_val.reset_index(drop=True), vl_es.reset_index(drop=True)], axis=1), num_iteration=model.best_iteration)
        
        optR = OptimizedRounder()
        optR.fit(val_pred, y_val)
        coefficients = optR.coefficients()
        pred_val_y_k = optR.predict(val_pred, coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_val_y_k)
        qwk_scores.append(qwk)
        print("QWK = ", qwk)
#         test_predictions_lgb1 = optR.predict(oof_test_lgb1.mean(axis=1), coefficients_).astype(np.int8)

        test_pred = model.predict(pd.concat([X_tst.reset_index(drop=True), ts_es.reset_index(drop=True)], axis=1), num_iteration=model.best_iteration)
        oof_train[valid_index] = val_pred
        oof_test[:, i] = test_pred
        i += 1
#         print("check test shape: ", X_test.shape, X_tst.shape, pd.concat([X_tst.reset_index(drop=True), ts_es.reset_index(drop=True)], axis=1).shape)

#     print("final shape: ",X_train.shape, X_test.shape)
    print('{} cv mean QWK score : {}'.format('LGBM', np.mean(qwk_scores)))
    return model, oof_train, oof_test#, oof_train_prob, oof_test_prob



















# Based on Bojan -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944
# and Nishant -> https://www.kaggle.com/nishkgp/more-improved-ridge-2-lgbm

import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import sys

###Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
###until Kaggle admins fix the wordbatch pip package installation
###sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
import re

NUM_BRANDS = 4500
NUM_CATEGORIES = 1200

develop = False
# develop= True

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])


def main():
    start_time = time.time()
    from time import gmtime, strftime
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # if 1 == 1:
    ###train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', engine='c')
    ###test = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv', engine='c')

    train = pd.read_table('../input/train.tsv', engine='c')
    test = pd.read_table('../input/test.tsv', engine='c')

    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    nrow_test = train.shape[0]  # -dftt.shape[0]
    dftt = train[(train.price < 1.0)]
    train = train.drop(train[(train.price < 1.0)].index)
    del dftt['price']
    nrow_train = train.shape[0]
    # print(nrow_train, nrow_test)
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, dftt, test])
    submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()

    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Split categories completed.'.format(time.time() - start_time))

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
    wb.dictionary_freeze= True
    X_name = wb.fit_transform(merge['name'])
    del(wb)
    X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    wb = CountVectorizer()
    X_category1 = wb.fit_transform(merge['general_cat'])
    X_category2 = wb.fit_transform(merge['subcat_1'])
    X_category3 = wb.fit_transform(merge['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    # wb= wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 3, "hash_ngrams_weights": [1.0, 1.0, 0.5],
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None})
                             , procs=8)
    wb.dictionary_freeze= True
    X_description = wb.fit_transform(merge['item_description'])
    del(wb)
    X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
    print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_name.shape)
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()

    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    #    pd.to_pickle((sparse_merge, y), "xy.pkl")
    # else:
    #    nrow_train, nrow_test= 1481661, 1482535
    #    sparse_merge, y = pd.read_pickle("xy.pkl")

    # Remove features with document frequency <=1
    print(sparse_merge.shape)
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    print(sparse_merge.shape)

    gc.collect()
    train_X, train_y = X, y
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)

    model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=50, inv_link="identity", threads=1)

    model.fit(train_X, train_y)
    print('[{}] Train FTRL completed'.format(time.time() - start_time))
    if develop:
        preds = model.predict(X=valid_X)
        print("FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

    predsF = model.predict(X_test)
    print('[{}] Predict FTRL completed'.format(time.time() - start_time))

    model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=4)

    model.fit(train_X, train_y)
    print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
    if develop:
        preds = model.predict(X=valid_X)
        print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

    predsFM = model.predict(X_test)
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

    params = {
        'learning_rate': 0.6,
        'application': 'regression',
        'max_depth': 4,
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'feature_fraction': 0.6,
        'nthread': 4,
        'min_data_in_leaf': 100,
        'max_bin': 31
    }

    # Remove features with document frequency <=100
    print(sparse_merge.shape)
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 100, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    print(sparse_merge.shape)

    train_X, train_y = X, y
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)

    d_train = lgb.Dataset(train_X, label=train_y)
    watchlist = [d_train]
    if develop:
        d_valid = lgb.Dataset(valid_X, label=valid_y)
        watchlist = [d_train, d_valid]

    model = lgb.train(params, train_set=d_train, num_boost_round=6000, valid_sets=watchlist, \
                      early_stopping_rounds=1000, verbose_eval=1000)

    if develop:
        preds = model.predict(valid_X)
        print("LGB dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

    predsL = model.predict(X_test)

    print('[{}] Predict LGB completed.'.format(time.time() - start_time))

    preds = (predsF * 0.2 + predsL * 0.3 + predsFM * 0.5)

    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_wordbatch_ftrl_fm_lgb.csv", index=False)


if __name__ == '__main__':
    main()