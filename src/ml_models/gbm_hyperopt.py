

import pandas as pd
import numpy as np
import os, gc
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from hyperopt import hp, tpe, fmin, Trials


def example_of_usuage():
    """
    example:
        best, tpe_trials = run_hyperopt_xgb(train_df, target, 3)
        best, tpe_trials = run_hyperopt_cat(train_df, target, 5)
        best, tpe_trials = run_hyperopt_lgb(train_df, target, 5)

        tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results], 
                                    'iteration': tpe_trials.idxs_vals[0]['num_leaves'],
                                    'x': tpe_trials.idxs_vals[1]['num_leaves']})
                                
    """
    print("more info can be find in tutorial (gridsearch-bayesian-opt)")




def train_hyperopt_lgb_model(X_train, y_train, X_valid, y_valid, features, param, num_round):
    """
    Args:
        X_train, X_valid: training and valid data
        y_train, y_valid: training and valid target
        features: training features
    Return:
        oof-pred, model,
    """
    _train = lgb.Dataset(X_train[features], label=y_train, feature_name=list(features))
    _valid = lgb.Dataset(X_valid[features], label=y_valid,feature_name=list(features))
    
    clf = lgb.train(param, _train, num_round, 
                    valid_sets = [_train, _valid], 
                    verbose_eval=False, 
                    early_stopping_rounds = 25)                  
    
    oof = clf.predict(X_valid[features], num_iteration=clf.best_iteration)

    return oof, clf
    

def run_hyperopt_lgb(train_df, target, max_evals):

    def bayesian_opt(param):
        random_seed = 1234
        n_splits = 3
        num_round = 10000

        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        oof_lgb = np.zeros(len(train_df))

        score_cv = []

        for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):

            y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
            X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
            features = X_train.columns

            num_round = 20
            oof, clf = train_hyperopt_lgb_model(X_train, y_train, 
                                                                    X_valid, y_valid, 
                                                                    features, param, 
                                                                    num_round)

            score = roc_auc_score(y_valid, oof)
            score_cv.append(score)
        
#         print(np.round(np.mean(score_cv), 4))
#         print( "  cv-auc = ", np.round(np.mean(score_cv), 4),'+/-', np.std(score_cv) )
        return -np.mean(score_cv)
    
    bayesian_params = {
        'bagging_freq'           : 5,
        'bagging_fraction'       : 0.33,
        'boost_from_average'     : 'false',
        'boost'                  : 'gbdt',
        'feature_fraction'       : hp.uniform('feature_fraction', 0.5, 0.8),
        'learning_rate'          : 0.01,#hp.uniform('learning_rate', 0.01, 0.3)
        'max_depth'              : -1,
        'metric'                 : 'auc',
        'min_data_in_leaf'       : hp.choice('min_data_in_leaf', np.linspace(50, 150, 5, dtype=int)),
        'min_sum_hessian_in_leaf': hp.uniform('min_sum_hessian_in_leaf', 0.1, 100),
        'num_leaves'             : hp.choice('num_leaves', np.arange(50, 90, dtype=int)),
        'num_threads'            : 4,
        'tree_learner'           : 'serial',
        'objective'              : 'binary',
        'verbosity'              : 1,
        'lambda_l1'              : hp.uniform('lambda_l1', 0.01, 5.0),
        'lambda_l2'              : hp.uniform('lambda_l2', 0.01, 5.0)
    }   

    
    trials = Trials()
    results = fmin(bayesian_opt, bayesian_params, algo=tpe.suggest, 
                   trials=trials, max_evals=max_evals)
        
    return results, trials 






def train_cat_model(X_train, y_train, X_valid, y_valid, features, param, num_round):
    """
    Args:
        X_train, X_valid: training and valid data
        y_train, y_valid: training and valid target
        features: training features
    Return:
        oof-pred, model
    """
    param['iterations'] = num_round
    
    _train = Pool(X_train[features], label=y_train)#, cat_features=cate_features_index)
    _valid = Pool(X_valid[features], label=y_valid)#, cat_features=cate_features_index)

    watchlist = [_train, _valid]
    clf = CatBoostClassifier(**param)
    clf.fit(_train, 
            eval_set=watchlist, 
            verbose=False,
            use_best_model=True)
        
    oof  = clf.predict_proba(X_valid[features])[:,1]

    return oof, clf


def run_hyperopt_cat(train_df, target, max_evals):

    def bayesian_opt(params):
        
        params['depth'] = int(params['depth'])
        n_splits = 3
        random_seed = 1234

        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        oof_cat = np.zeros(len(train_df))
        score_cv = []

        ##########################
        for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):

            y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
            X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
            features = X_train.columns

            num_rounds = 20
            oof, clf = train_cat_model(X_train, y_train, 
                                       X_valid, y_valid, 
                                       features, params, 
                                       num_rounds)

            oof_cat[valid_index] = oof
            score = roc_auc_score(y_valid, oof)
            score_cv.append(score)
    
#         print(np.round(np.mean(score_cv), 4))
#         print( "  cv-auc = ", np.round(np.mean(score_cv), 4),'+/-', np.std(score_cv) )
        return -np.mean(score_cv)
    
    
    bayesian_params = {
        'loss_function'         : "Logloss", 
        'eval_metric'           : "AUC",
        'random_strength'       : hp.quniform('random_strength', 1, 3, 0.2),
        'border_count'          : 128,
        'scale_pos_weight'      : 3.507,
        'depth'                 : hp.quniform('depth', 4, 7, 1),
        #hp.choice('depth', [4,5,6,7]), 
        'early_stopping_rounds' : 50,
        'random_seed'           : 1337,
        'task_type'             : 'CPU', 
        # subsample can't be used, while bootstrap (bayesian) is used.
#         'subsample'             : hp.quniform('subsample', 0.3, 0.8, 0.1), 
#         'iterations'            : 10000, 
        'learning_rate'         : 0.09,
        'thread_count'          : 4,
        'l2_leaf_reg'           : 3,
#         'grow_policy'           : hp.choice('grow_policy', 
#                                             ['SymmetricTree', 'Depthwise', 'Lossguide']),
#         'min_data_in_leaf'      : hp.choice('min_data_in_leaf', np.arange(50, 150, 5)),
        
#         hp.choice('min_data_in_leaf', [
#             {'grow_policy': 'SymmetricTree', 'min_data_in_leaf': None},
#             {'grow_policy': ['Depthwise', 'Lossguide'], 
#              'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(50, 150, 5))},
#             ])
    }
    
    
    
    trials = Trials()
    results = fmin(bayesian_opt, bayesian_params, algo=tpe.suggest, 
                   trials=trials, max_evals=max_evals)
        
    return results, trials 





def train_xgb_model(X_train, y_train, X_valid, y_valid, features, param, X_test, 
                    num_round):
    """
    Args:
        X_train, X_valid: training and valid data
        y_train, y_valid: training and valid target
        features: training features
    Return:
        oof-pred, test_preds, model, model_imp
    """
    _train = xgb.DMatrix(X_train[features], label=y_train, feature_names=list(features))
    _valid = xgb.DMatrix(X_valid[features], label=y_valid,feature_names=list(features))
    
    watchlist = [(_valid, 'valid')]
    clf = xgb.train(dtrain=_train, 
                    num_boost_round=num_round, 
                    evals=watchlist,
                    early_stopping_rounds=25, 
                    verbose_eval=False, 
                    params=param)
    
    valid_frame = xgb.DMatrix(X_valid[features],feature_names=list(features))
    oof  = clf.predict(valid_frame, ntree_limit=clf.best_ntree_limit)


    test_frame = xgb.DMatrix(X_test[features],feature_names=list(features))
    test_pred = clf.predict(test_frame, ntree_limit=clf.best_ntree_limit)

    
    xgb_imp = pd.DataFrame(data=[list(clf.get_fscore().keys()), 
                                 list(clf.get_fscore().values())]).T
    xgb_imp.columns = ['feature','imp']
    xgb_imp.imp = xgb_imp.imp.astype('float')
    
    return oof, test_pred, clf, xgb_imp



def run_hyperopt_xgb(train_df, target, max_evals):

    def bayesian_opt_xgb(params):

        params['max_depth'] = int(params['max_depth'])
        features = train_df.columns
        n_splits = 3
        random_seed = 1234
        feature_imp = pd.DataFrame()

        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        oof_xgb = np.zeros(len(train_df))
        score_cv = []
        
    ##########################
        for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):

            y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
            X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
            features = X_train.columns


            num_rounds = 10000
            oof, test_pred, clf, xgb_imp = train_xgb_model(X_train, y_train, 
                                                           X_valid, y_valid, 
                                                           features, params, 
                                                           test_df, num_rounds)

            oof_xgb[valid_index] = oof
            score = roc_auc_score(y_valid, oof)
            score_cv.append(score)
    
#         print(np.round(np.mean(score_cv), 4))
#         print( "  cv-auc = ", np.round(np.mean(score_cv), 4),'+/-', np.std(score_cv) )
        return -np.mean(score_cv)
    
    bayesian_params = {
        'eval_metric'     : 'auc',
        'seed'            : 1337,
        'eta'             : 0.05,#hp.quniform('learning_rate', 0.005, 0.03, 0.005)
        'subsample'       : hp.quniform('subsample', 0.3, 0.8, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 0.7, 0.05),
        'silent'          : 1,
        'nthread'         : 4,
        'Scale_pos_weight': 3.607,
        'objective'       : 'binary:logistic',
        'max_depth'       : hp.quniform('max_depth', 3, 5, 1),
        'alpha'           : hp.quniform('alpha', 0.01, 3, 0.05),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'gamma'           : hp.quniform('gamma', 0.5, 10, 0.05)
    }
    

    
    trials = Trials()
    results = fmin(bayesian_opt_xgb, bayesian_params, algo=tpe.suggest, 
                   trials=trials, max_evals=max_evals)
        
    return results, trials 

