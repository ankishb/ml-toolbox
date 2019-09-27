from catboost import Pool, CatBoostRegressor, CatBoostClassifier

cat_params_class = { 'depth': 5, 'eta': 0.2748, 'task_type':"GPU", 'random_strength': 1.5809,#, 'num_rounds':20000
        'reg_lambda': 6,'od_type': 'Iter','border_count': 128,#'metrics': 'Logloss', 
        'bootstrap_type' : "Bayesian",'random_seed': 123455,'verbose_eval': 200,'early_stopping_rounds': 100, 
        'num_boost_round': 10000,  'loss_function':'MultiClass',}

def run_cat_class(params, X_train, X_test):
    n_splits = 5#12
    from sklearn.model_selection import StratifiedKFold, GroupKFold
#     kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)
    kfold = GroupKFold(n_splits=n_splits)#, random_state=1337)
    oof_train = np.zeros((X_train.shape[0], 5))
    oof_test = np.zeros((X_test.shape[0], 5, n_splits))
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
        cat_model = CatBoostClassifier(**params)#loss_function="Logloss", eval_metric="AUC"

        cat_model.fit(d_train, eval_set=d_valid, verbose=verbose_eval,use_best_model=True)

        val_pred = cat_model.predict_proba(X_val)
        test_pred = cat_model.predict_proba(X_test)
        oof_train[valid_index,:] = val_pred
        oof_test[:, :, i] = test_pred     
        i += 1     
#     print('{} cv mean QWK score : {}'.format('Cat', np.mean(qwk_scores)))
    return cat_model, oof_train, oof_test#, oof_train_prob, oof_test_prob




import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold

lgb_params_class = {'application': 'regression',
                  'boosting': 'gbdt',
                  'metric': 'multi_logloss',
                  'num_leaves': 70,
                  'objective': 'multiclass',
                  'max_depth': 9,
                  'learning_rate': 0.01,
                  'bagging_fraction': 0.85,
                  'feature_fraction': 0.8,
                  'min_split_gain': 0.02,
                  'min_child_samples': 150,
                  'min_child_weight': 0.02,
                  'lambda_l2': 0.0475,
                  'verbosity': -1,
                  'data_random_seed': 17,
                  'num_machines':4,
                  'num_class':5}

def run_lgb_class(params, X_train_, X_test_, cat_cols=None):
    
    X_train, X_test = X_train_, X_test_
    # Additional parameters:
    early_stop = 100
    verbose_eval = 100
    num_rounds = 10000
    n_splits = 5
    
    feature_importance_df = pd.DataFrame()

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
#     for train_index, valid_index in stratified_group_k_fold(X=X_train, y=X_train['AdoptionSpeed'].astype('int64').values, 
#                                                             groups= np.array(X_train.RescuerID.values), 
#                                                             k=5, seed=2019):
        """stratified_group_k_fold(X, y, groups, k, seed=None)"""
#         train_index = shuffle(train_index, random_state=2019)
        print("fold ",i)
        X_tr, X_val = X_train.iloc[train_index, :], X_train.iloc[valid_index, :]

        y_tr, y_val = X_tr['AdoptionSpeed'].values, X_val['AdoptionSpeed'].values
        X_tr, X_val = X_tr.drop(['AdoptionSpeed'], axis=1), X_val.drop(['AdoptionSpeed'], axis=1)


        if cat_cols is None:
            d_train = lgb.Dataset(X_tr, label=y_tr)#, categorical_feature=cat_cols)
            d_valid = lgb.Dataset(X_val, label=y_val)#, categorical_feature=cat_cols)
        else:
            d_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
            d_valid = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)
        watchlist = [d_train, d_valid]

        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        oof_train[valid_index, :] = val_pred
        oof_test[:, :, i] = test_pred
        rmse_scores.append(model.best_score['valid_1']['multi_logloss'])
        i += 1
        
#     oof_test = off_test.mean(axis=-1)
    print('{} cv LogLoss mean error: {}'.format('LGBM', np.mean(rmse_scores)))

    return model, oof_train, oof_test#, oof_train_prob, oof_test_prob

