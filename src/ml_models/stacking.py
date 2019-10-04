
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import classification_report as cr
import xgboost as xgb

import numpy as np
import pandas as pd
import gc

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC


def xgb_model(X_train, y_train, X_valid, y_valid, features, X_test, 
              num_round=10000, depth=4):
    param = {
        'eval_metric'     : 'auc',
        'seed'            : 1337,
        'eta'             : 0.05,
        'subsample'       : 0.7,
        'colsample_bytree': 0.5,
        'silent'          : 1,
        'nthread'         : 4,
        'Scale_pos_weight': 100,#3.607,
        'objective'       : 'binary:logistic',
        'max_depth'       : depth,
        'alpha'           : 0.09
    }
        
    _train = xgb.DMatrix(
        X_train[features], label=y_train, 
        feature_names=list(features)
    )
    _valid = xgb.DMatrix(
        X_valid[features], label=y_valid, 
        feature_names=list(features)
    )
    
    watchlist = [(_valid, 'valid')]
    clf = xgb.train(dtrain=_train, 
                    num_boost_round=num_round, 
                    evals=watchlist,
                    early_stopping_rounds=25, 
                    verbose_eval=0, 
                    params=param)
    
    valid_frame = xgb.DMatrix(X_valid[features],feature_names=list(features))
    oof  = clf.predict(valid_frame, ntree_limit=clf.best_ntree_limit)


    test_frame = xgb.DMatrix(X_test[features],feature_names=list(features))
    test_pred = clf.predict(test_frame, ntree_limit=clf.best_ntree_limit)
    
    return oof, test_pred


def passive_agg_model(X_train, y_train, X_valid, y_valid, X_test, 
                      alpha=1, balance='balanced'):

    clf = PassiveAggressiveClassifier(
            C=alpha, fit_intercept=True, max_iter=None, tol=None, 
#             early_stopping=False, validation_fraction=0.2, n_iter_no_change=5, 
            shuffle=True, verbose=0, n_jobs=-1, random_state=1234, loss='squared_hinge',
            class_weight=balance, average=False, n_iter=500)
    
    means = X_train.mean()
    clf.fit(X_train.fillna(means), y_train)
    oof  = clf.predict(X_valid.fillna(means))
    test_pred = clf.predict(X_test.fillna(means))

    return oof, test_pred

def logreg_model(X_train, y_train, X_valid, y_valid, X_test, 
                 alpha=500, balance='balanced'):
    
    clf = LogisticRegression(
            penalty='l2', dual=False, C=alpha, fit_intercept=True, 
            intercept_scaling=1, class_weight=balance, random_state=1234, 
            max_iter=100, verbose=0, n_jobs=-1)
    
    means = X_train.mean()
    clf.fit(X_train.fillna(means), y_train)
    oof  = clf.predict_proba(X_valid.fillna(means))[:,1]
    test_pred = clf.predict_proba(X_test.fillna(means))[:,1]

    return oof, test_pred

def ridge_model(X_train, y_train, X_valid, y_valid, X_test, 
                 alpha=0.01, balance='balanced'):
    
    clf = RidgeClassifier(
        alpha=alpha, fit_intercept=True, normalize=True, 
        class_weight=balance, random_state=1234)
    
    means = X_train.mean()
    clf.fit(X_train.fillna(means), y_train)
    oof  = clf.predict(X_valid.fillna(means))
    test_pred = clf.predict(X_test.fillna(means))

    return oof, test_pred
    
def stacking(train_df, target, test_df, split=4, depth=4, print_=False):

    n_splits = split
    random_seed = 1234
    
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    oof_cat = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_pas = np.zeros(len(train_df))
    oof_log = np.zeros(len(train_df))
    oof_rid = np.zeros(len(train_df))
    
    pred_cat = np.zeros((len(test_df),n_splits))
    pred_xgb = np.zeros((len(test_df),n_splits))
    pred_pas = np.zeros((len(test_df),n_splits))
    pred_log = np.zeros((len(test_df),n_splits))
    pred_rid = np.zeros((len(test_df),n_splits))
    
    f1s = []
##########################
    for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):
        if print_: print("Fold {}".format(fold_), end=" \t\t")
    
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
        features = X_train.columns
#         if print_: print(X_train.shape, X_valid.shape)
        
        num_rounds = 10000
        ####################################
        # cat model
        ####################################
        oof, test_pred = cat_model(
            X_train, y_train, 
            X_valid, y_valid, 
            features,test_df, 
            num_round=num_rounds,
            depth=depth
        )
    
        oof_cat[valid_index] = oof
        pred_cat[:,fold_] = test_pred
        
        y_bin = [1. if y_cont > 0.5 else 0. for y_cont in oof]
        cat_score = f1_score(y_valid, y_bin, average='weighted')
        gc.collect()

        #################################
        # xgb model
        #################################
        oof, test_pred = xgb_model(
            X_train, y_train, 
            X_valid, y_valid, 
            features, test_df, 
            num_round=num_rounds,
            depth=depth
        )
        
        oof_xgb[valid_index] = oof
        pred_xgb[:,fold_] = test_pred
        
        y_bin = [1. if y_cont > 0.5 else 0. for y_cont in oof]
        xgb_score = f1_score(y_valid, y_bin, average='weighted')
#         if print_: print("="*60)
        f1s.append(xgb_score)

        ####################################
        # passive regressive
        ####################################
        oof, test_pred = passive_agg_model(
            X_train, y_train, 
            X_valid, y_valid, 
            test_df, alpha=1
        )
        
        oof_pas[valid_index] = oof
        pred_pas[:,fold_] = test_pred
        
        y_bin = [1. if y_cont > 0.5 else 0. for y_cont in oof]
        pas_score = f1_score(y_valid, y_bin, average='weighted')

        ####################################
        # logistic regression
        ####################################
        oof, test_pred = logreg_model(
            X_train, y_train, 
            X_valid, y_valid, 
            test_df, alpha=1
        )
        
        oof_log[valid_index] = oof
        pred_log[:,fold_] = test_pred
        
        y_bin = [1. if y_cont > 0.5 else 0. for y_cont in oof]
        log_score = f1_score(y_valid, y_bin, average='weighted')

        ####################################
        # ridge regression
        ####################################
        oof, test_pred = ridge_model(
            X_train, y_train, 
            X_valid, y_valid, 
            test_df, alpha=1
        )
        
        oof_rid[valid_index] = oof
        pred_rid[:,fold_] = test_pred
        
        y_bin = [1. if y_cont > 0.5 else 0. for y_cont in oof]
        rid_score = f1_score(y_valid, y_bin, average='weighted')
        ####################################
        
        if print_:
            print("CAT: {:.3f} \t XGB: {:.3f} \t PASS-AGG: {:.3f} \t LOG-REG: {:.3f} \t RIDGE: {:.3f}".format(
                cat_score, xgb_score, pas_score, log_score, rid_score
            ))

        
    print("="*60)
    print( "  mean-f1 = ", np.mean(f1s))
    print("="*60)

    oofs = np.stack([oof_cat, oof_xgb, oof_pas, oof_log, oof_rid]).T
    preds = np.stack([pred_cat.mean(1), pred_xgb.mean(1), pred_pas.mean(1), pred_log.mean(1), pred_rid.mean(1)]).T
    return oofs, preds