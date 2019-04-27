








from catboost import Pool, CatBoostRegressor, CatBoostClassifier
def run_cat_class_hyper_opt(param):

    X_train, X_test = X_train_non_null, X_test_non_null

    eta                     = '{:.4f}'.format(param['eta'])
    random_strength         = '{:.4f}'.format(param['random_strength'])
    l2_leaf_reg             = '{:.4f}'.format(param['l2_leaf_reg'])
#     reg_lambda              = '{:.4f}'.format(param['reg_lambda'])
    depth                   = int(param['depth'])

    eta                    = np.float(eta)
    random_strength        = np.float(random_strength)
    l2_leaf_reg            = np.float(l2_leaf_reg)
    depth                  = np.int(depth)
    """
    od_type: to detct overfitting
        IncToDec
        Iter
    """


    params = { 
    'depth'                  : 3,#6, 
    'eta'                    : 0.076,#0.01, 
    'task_type'              :"GPU", 
    'random_strength'        : 0.8157,#1.5,
    'loss_function'          : 'MultiClass', 
#     'reg_lambda'             : reg_lambda,
    'l2_leaf_reg'            : 1.53,
    'od_type'                : 'Iter',
    'border_count'           : 128,#border_count=254 for best quality on GPU
    'bootstrap_type'         : "Bayesian",
    'random_seed'            : 123455,
    'verbose_eval'           : 200,
    'early_stopping_rounds'  : 100, 
    'num_boost_round'        : 10000
    }



    n_splits = 5#12
    from sklearn.model_selection import StratifiedKFold, GroupKFold
#     kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)
    kfold = GroupKFold(n_splits=n_splits)#, random_state=1337)
    # oof_train = np.zeros((X_train.shape[0]))
    # oof_test = np.zeros((X_test.shape[0], n_splits))
    qwk_scores = []
    i = 0
    
#     num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
        
#     for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values):
    for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values, X_train.RescuerID.values):
        print("fold ",i)
        X_tr, X_val = X_train.iloc[train_index, :], X_train.iloc[valid_index, :]
        y_tr  = X_tr['AdoptionSpeed']#.values
        y_val = X_val['AdoptionSpeed'].values
        X_tr, X_val = X_tr.drop(['AdoptionSpeed'], axis=1), X_val.drop(['AdoptionSpeed'], axis=1)
        
        y_tr = y_tr.values
        y_tr_ = y_tr# + 0.15*np.random.randn(y_tr.shape[0])
#         if cat_feat is None:
        d_train = Pool(X_tr, label=y_tr_)#, categorical_feature=cat_feat)
        d_valid = Pool(X_val, label=y_val)#, categorical_feature=cat_feat)

        watchlist = [d_train, d_valid]
        cat_model = CatBoostClassifier(**params)
        cat_model.fit(d_train, eval_set=d_valid, verbose=verbose_eval,use_best_model=True)
                              
        # val_pred = cat_model.predict(X_val)

        qwk_scores.append(cat_model.get_best_score()['validation_0']['MultiClass'] )
        i += 1
    return -np.mean(qwk_scores)




def run_lgb_class_hyper_opt(param):

    X_train, X_test = X_train_non_null, X_test_non_null
    # LightGBM expects next three parameters need to be integer. So we make them integer
    num_leaves              = int(param['num_leaves'])
#     min_data_in_leaf        = int(param['min_data_in_leaf'])
    max_bin                 = int(param['max_bin'])
#     learning_rate           = '{:.3f}'.format(param['learning_rate'])
#     min_sum_hessian_in_leaf = '{:.5f}'.format(param['min_sum_hessian_in_leaf'])
#     lambda_l1               = '{:.3f}'.format(param['lambda_l1'])
#     lambda_l2               = '{:.3f}'.format(param['lambda_l2'])
#     min_gain_to_split       = '{:.3f}'.format(param['min_gain_to_split'])
    feature_fraction        = '{:.3f}'.format(param['feature_fraction'])
#     max_depth               = int(param['max_depth'])

#     '{:.3f}'.format(params['colsample_bytree'])
    assert type(num_leaves)       == int
#     assert type(min_data_in_leaf) == int
    assert type(max_bin)          == int 
#     assert type(max_depth) == int

    # feature_fraction       = 0.568
    lambda_l1              = 1.44
    lambda_l2              = 1.1527
    # max_bin                = 88
    # min_data_in_leaf       = 41
    # num_leaves             = 72


    params = {'application': 'regression',
        'boosting'         : 'gbdt',
        'metric'           : 'multi_logloss',
        'num_leaves'       : num_leaves,#70,
        'max_depth'        : 7,
        'objective'        : 'multiclass',
        'learning_rate'    : 0.03,
        'bagging_fraction' : 0.85,
        'feature_fraction' : feature_fraction,#0.5684914054241338,#0.8,
        'min_split_gain'   : 0.02,
        'min_child_samples': 150,
        'min_child_weight' : 0.02,
        'lambda_l2'        : lambda_l2,#0.0475,
        'lambda_l1'        : lambda_l1,
        'verbosity'        : -1,
        'data_random_seed' : 17,
        'num_machines'     : 4,
        'max_bin'          : max_bin,
        'num_class'        : 5,
#         'min_data_in_leaf' : min_data_in_leaf,
        }
    # Additional parameters:
    import time
    start_time_ = time.time()
    early_stop = 500
    verbose_eval = 100
    num_rounds = 10000
    n_splits = 3

    from sklearn.model_selection import StratifiedKFold, GroupKFold
#     kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)
    kfold = GroupKFold(n_splits=n_splits)
    oof_train = np.zeros((X_train.shape[0], 5))
    oof_test = np.zeros((X_test.shape[0], 5, n_splits))
    qwk_scores = []
    rmse_scores = []
    i = 0
    print("running for {} splits".format(n_splits))
#     for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values):
    for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values, X_train.RescuerID):
        print("fold ",i)
        X_tr, X_val = X_train.iloc[train_index, :], X_train.iloc[valid_index, :]

        y_tr, y_val = X_tr['AdoptionSpeed'].values, X_val['AdoptionSpeed'].values
        X_tr, X_val = X_tr.drop(['AdoptionSpeed'], axis=1), X_val.drop(['AdoptionSpeed'], axis=1)

        d_train = lgb.Dataset(X_tr, label=y_tr)
        d_valid = lgb.Dataset(X_val, label=y_val)
        watchlist = [d_train, d_valid]

        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

#         val_pred = model.predict(X_val, num_iteration=model.best_iteration)
#         test_pred = model.predict(X_test, num_iteration=model.best_iteration)
#         oof_train[valid_index,:] = val_pred
#         oof_test[:, :, i] = test_pred
        i += 1
#         print("time taken: ",(time.time()-start_time_)/60)
        rmse_scores.append(model.best_score['valid_1']['multi_logloss'])
    return np.mean(rmse_scores)

from skopt import BayesSearchCV
from hyperopt import hp, tpe, fmin
import time
st = time.time()

    # feature_fraction       = 0.568
    # lambda_l1              = 1.44
    # lambda_l2              = 1.1527
    # max_bin                = 88
    # min_data_in_leaf       = 41
    # num_leaves             = 72

space = {
    'num_leaves':              hp.quniform('num_leaves', 50, 90, 1), 
#     'min_data_in_leaf':        hp.uniform('min_data_in_leaf', 40, 90),  
#     'learning_rate':           hp.uniform('learning_rate', 0.01, 0.3),
#     'min_sum_hessian_in_leaf': 0.004104757877722484,#hp.uniform('min_sum_hessian_in_leaf', 0.00001, 0.01),    
    'feature_fraction':        hp.uniform('feature_fraction', 0.4, 0.8),
#     'lambda_l1':               hp.uniform('lambda_l1', 0.5, 3.0), 
#     'lambda_l2':               hp.uniform('lambda_l2', 0.5, 3.0), 
#     'min_gain_to_split':       hp.quniform('min_gain_to_split', 0, 1.0, 0.2),
    # 'min_data_in_leaf':       hp.quniform('min_data_in_leaf', 0, 1.0, 0.2),
#     'max_depth':               6#hp.quniform('max_depth', 3,15,2),
    'max_bin':                 hp.uniform('max_bin', 60, 110)
}

#     gamma                   = '{:.4f}'.format(param['gamma'])
#     alpha                   = '{:.4f}'.format(param['alpha'])
#     lambd                  = '{:.4f}'.format(param['lambda'])
#     subsample               = '{:.4f}'.format(param['subsample'])
#     feature_fraction        = '{:.3f}'.format(param['feature_fraction'])
#     colsample_bytree        = '{:.3f}'.format(param['colsample_bytree'])
#     max_depth               = int(param['max_depth'])
# space = {
# #     'num_leaves':              hp.uniform('num_leaves', 50, 90), 
# #     'min_data_in_leaf':        hp.uniform('min_data_in_leaf', 40, 50),  
# #     'learning_rate':           hp.uniform('learning_rate', 0.01, 0.3),
# #     'min_sum_hessian_in_leaf': 0.004104757877722484,#hp.uniform('min_sum_hessian_in_leaf', 0.00001, 0.01),    
#     'feature_fraction':        hp.uniform('feature_fraction', 0.3, 0.8),
#     'lambda':                  hp.uniform('lambda', 0.1, 3.0), 
#     'alpha':                   hp.uniform('alpha', 0.01, 3.0), 
#     'colsample_bytree':        hp.uniform('colsample_bytree', 0, 0.9),
#     'gamma':                   hp.uniform('gamma', 0, 50),
#     'max_depth':               hp.quniform('max_depth', 3,8,1),
#     'subsample':               hp.uniform('subsample', 0.3, 1)
# }

    
# space = {
#     'eta':                     hp.uniform('eta', 0.01, 0.3),
#     'random_strength':         hp.uniform('random_strength', 0.5, 3),
#     'l2_leaf_reg':             hp.uniform('l2_leaf_reg', 0.1, 3.0), 
# #     'reg_lambda':              hp.uniform('reg_lambda', 0.01, 5.0), 
#     'depth':                   hp.quniform('depth', 2, 7, 1),
# }
best = fmin(fn=run_lgb_class_hyper_opt,
            space=space,
            algo=tpe.suggest,
            max_evals=25)
end = time.time()
print("total time taken: ",(end-start)/60)

xgb_params =  {
'alpha': 0.7129825122251625, 
'colsample_bytree': 0.6966290443207815, 
'feature_fraction': 0.5669645377388706, 
'gamma': 49.80629641831484, 
'lambda': 1.8005881371183308, 
'max_depth': 4.0, 
'subsample': 0.9288690472308233
}
lgb_params = {
 'feature_fraction': 0.5684914054241338,
 'lambda_l1': 1.4406582960652417,
 'lambda_l2': 1.1527248885155075,
 'max_bin': 88.24384833755381,
 'min_data_in_leaf': 41.254358809229345,
 'num_leaves': 72.22613521249644
}

cat_params =  {
 'depth': 3.9899936637938342, 
 'eta': 0.0763211669306367, 
 'l2_leaf_reg': 1.5311893083273935, 
 'random_strength': 0.8157594408296202
}

cat_class_params:  {
 'depth': 5.0, 
 'eta': 0.27482002333885297, 
 'l2_leaf_reg': 1.5494118792295957, 
 'random_strength': 1.5809731466418282
}



"""

For Faster Speed

    Use bagging by setting bagging_fraction and bagging_freq
    Use feature sub-sampling by setting feature_fraction
    Use small max_bin
    Use save_binary to speed up data loading in future learning
    Use parallel learning, refer to Parallel Learning Guide

For Better Accuracy

    Use large max_bin (may be slower)
    Use small learning_rate with large num_iterations
    Use large num_leaves (may cause over-fitting)
    Use bigger training data
    Try dart

Deal with Over-fitting

    Use small max_bin
    Use small num_leaves
    Use min_data_in_leaf and min_sum_hessian_in_leaf
    Use bagging by set bagging_fraction and bagging_freq
    Use feature sub-sampling by set feature_fraction
    Use bigger training data
    Try lambda_l1, lambda_l2 and min_gain_to_split for regularization
    Try max_depth to avoid growing deep tree

"""


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold

def run_lgb_hyper_opt(param):

    X_train, X_test = X_train_non_null, X_test_non_null
    # LightGBM expects next three parameters need to be integer. So we make them integer
    num_leaves              = int(param['num_leaves'])
    min_data_in_leaf        = int(param['min_data_in_leaf'])
    max_bin                 = int(param['max_bin'])
#     learning_rate           = '{:.3f}'.format(param['learning_rate'])
#     min_sum_hessian_in_leaf = '{:.5f}'.format(param['min_sum_hessian_in_leaf'])
    lambda_l1               = '{:.3f}'.format(param['lambda_l1'])
    lambda_l2               = '{:.3f}'.format(param['lambda_l2'])
#     min_gain_to_split       = '{:.3f}'.format(param['min_gain_to_split'])
    feature_fraction        = '{:.3f}'.format(param['feature_fraction'])
#     max_depth               = int(param['max_depth'])

#     '{:.3f}'.format(params['colsample_bytree'])
    assert type(num_leaves)       == int
    assert type(min_data_in_leaf) == int
    assert type(max_bin)          == int 
#     assert type(max_depth) == int

    feature_fraction       = 0.568
    lambda_l1              = 1.44
    lambda_l2              = 1.1527
    max_bin                = 88
    min_data_in_leaf       = 41
    num_leaves             = 72

    params = {'application': 'regression',
        'boosting'         : 'gbdt',
        'metric'           : 'rmse',
        'num_leaves'       : num_leaves,#70,
        'max_depth'        : 7,
        'learning_rate'    : 0.01,
        'bagging_fraction' : 0.85,
        'feature_fraction' : feature_fraction,#0.8,
        'min_split_gain'   : 0.02,
        'min_child_samples':150,
        'min_child_weight' : 0.02,
        'lambda_l2'        : lambda_l2,#0.0475,
        'lambda_l1'        : lambda_l1,
        'verbosity'        : -1,
        'data_random_seed' : 17,
        'num_machines'     : 4,
        'max_bin'          : max_bin,
        # 'min_data_in_leaf': min_data_in_leaf
        }

    # Additional parameters:
    early_stop = 500
    verbose_eval = 100
    num_rounds = 10000
    n_splits = 3

    from sklearn.model_selection import StratifiedKFold, GroupKFold
#     kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)
    kfold = GroupKFold(n_splits=n_splits)
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    qwk_scores = []
    rmse_scores = []
    i = 0
    print("running for {} splits".format(n_splits))
#     for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values):
    for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values, X_train.RescuerID):
        print("fold ",i)
        X_tr, X_val = X_train.iloc[train_index, :], X_train.iloc[valid_index, :]

        y_tr, y_val = X_tr['AdoptionSpeed'].values, X_val['AdoptionSpeed'].values
        X_tr, X_val = X_tr.drop(['AdoptionSpeed'], axis=1), X_val.drop(['AdoptionSpeed'], axis=1)

#         print('\ny_tr distribution: {}'.format(Counter(y_tr)))
        d_train = lgb.Dataset(X_tr, label=y_tr)
        d_valid = lgb.Dataset(X_val, label=y_val)
        watchlist = [d_train, d_valid]

        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        oof_train[valid_index] = val_pred
        oof_test[:, i] = test_pred
        i += 1
        
        optR = OptimizedRounder()
        optR.fit(val_pred, y_val)
        coefficients = optR.coefficients()
        valid_pred = optR.predict(val_pred, coefficients)
        qwk = quadratic_weighted_kappa(y_val, valid_pred)
        print("lgb QWK = ", qwk)
        qwk_scores.append(qwk)
        rmse_scores.append(model.best_score['valid_1']['rmse'])

    try:
        print('{} cv RMSE score : {} and their mean: {}'.format('LGBM', rmse_scores, np.mean(rmse_scores)))
    except:
        pass
    return -np.mean(qwk_scores)








import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

"""
Control Overfitting

When you observe high training accuracy, but low test accuracy, it is likely that you encountered overfitting problem.

There are in general two ways that you can control overfitting in XGBoost:

    The first way is to directly control model complexity.
        This includes max_depth, min_child_weight and gamma.
    The second way is to add randomness to make training robust to noise.
        This includes subsample and colsample_bytree.
        You can also reduce stepsize eta. Remember to increase num_round when you do so.

"""

def run_xgb_hyper_opt(params):
	X_train, X_test = X_train_non_null, X_test_non_null
    # LightGBM expects next three parameters need to be integer. So we make them integer
    # num_leaves              = int(param['num_leaves'])
    # min_child_weight        = int(param['min_child_weight'])
    # max_bin                 = int(param['max_bin'])
    # eta                     = '{:.4f}'.format(param['eta'])
    gamma                   = '{:.4f}'.format(param['gamma'])
    alpha                   = '{:.4f}'.format(param['alpha'])
    lambd                  = '{:.4f}'.format(param['lambda'])
    subsample               = '{:.4f}'.format(param['subsample'])
    feature_fraction        = '{:.3f}'.format(param['feature_fraction'])
    colsample_bytree        = '{:.3f}'.format(param['colsample_bytree'])
    max_depth               = int(param['max_depth'])

#     '{:.3f}'.format(params['colsample_bytree'])
    # assert type(num_leaves)       == int
    # assert type(min_data_in_leaf) == int
    # assert type(max_bin)          == int 
#     assert type(max_depth) == int




    eta = 0.0123
	params = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'eta': eta,#0.0123,
    'subsample': subsample,#0.8,
    'colsample_bytree': colsample_bytree,#0.85,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': 1,
    'alpha':alpha,#1,
    'lambda':lambd,#1,
    'max_depth':max_depth,
    'gamma': gamma
	}

    n_splits = 3
    verbose_eval = 1000
    num_rounds = 60000
    early_stop = 500

    from sklearn.model_selection import StratifiedKFold, GroupKFold
#     kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
    kf = GroupKFold(n_splits=n_splits)

    # oof_train = np.zeros((X_train.shape[0]))
    # oof_test = np.zeros((X_test.shape[0], n_splits))

	qwk_scores = []
    i = 0
    print("running for {} splits".format(n_splits))
#     for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):
    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values, X_train.RescuerID):
        print("fold ",i)
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)        
        # test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        valid_pred = optR.predict(valid_pred, coefficients)
        qwk = quadratic_weighted_kappa(y_val, valid_pred)
        print("xgb QWK = ", qwk)
        qwk_scores.append(qwk)
        # rmse_scores.append(model.best_score['valid_1']['rmse'])
        # oof_train[valid_idx] = valid_pred
        # oof_test[:, i] = test_pred
        i += 1
    return -np.mean(qwk_scores)





from skopt import BayesSearchCV
from hyperopt import hp, tpe, fmin

# space = {
#     'n_estimators': hp.quniform('n_estimators', 25, 500, 25),
#     'max_depth': hp.quniform('max_depth', 1, 10, 1)
# }
import time
st = time.time()
space = {
    'num_leaves':              hp.uniform('num_leaves', 50, 90), 
    'min_data_in_leaf':        hp.uniform('min_data_in_leaf', 40, 50),  
#     'learning_rate':           hp.uniform('learning_rate', 0.01, 0.3),
#     'min_sum_hessian_in_leaf': 0.004104757877722484,#hp.uniform('min_sum_hessian_in_leaf', 0.00001, 0.01),    
    'feature_fraction':        hp.uniform('feature_fraction', 0.5, 0.8),
    'lambda_l1':               hp.uniform('lambda_l1', 0.1, 3.0), 
    'lambda_l2':               hp.uniform('lambda_l2', 0.01, 2.0), 
#     'min_gain_to_split':       hp.quniform('min_gain_to_split', 0, 1.0, 0.2),
    # 'min_data_in_leaf':       hp.quniform('min_data_in_leaf', 0, 1.0, 0.2),
#     'max_depth':               6#hp.quniform('max_depth', 3,15,2),
    'max_bin':                 hp.uniform('max_bin', 40, 150)
}


best = fmin(fn=LGB_bayesian,
            space=space,
            algo=tpe.suggest,
            max_evals=25)
end = time.time()



    trials = Trials()
    results = fmin(
        objective, param_grid, algo=tpe.suggest,
        trials=trials, max_evals=max_evals)
    
    return results, trials.results 
