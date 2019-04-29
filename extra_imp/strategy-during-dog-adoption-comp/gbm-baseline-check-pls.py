

file_path = 'baseline'
random_seed = 2019
splits = 3
sub = sub_df.copy()
del sub_df





from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

features = train_df.columns
params = {
    'eval_metric'     : 'auc',
    'seed'            : 1337,
    'eta'             : 0.05,
    'subsample'       : 0.7,
    'colsample_bytree': 0.5,
    'silent'          : 1,
    'nthread'         : 4,
    'Scale_pos_weight': 3.607,
    'objective'       : 'binary:logistic',
    'max_depth'       : 4,
    'alpha'           : 0.05
}

n_splits = splits
verbose_eval = 50
early_stop = 50
num_rounds = 10000


folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
oof_xgb = np.zeros(len(train_df))
test_pred = np.zeros((len(test_df),n_splits))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    
    d_train = xgb.DMatrix( data          = train_df.iloc[trn_idx][features], 
                           label         = target.iloc[trn_idx], 
                           feature_names = train_df.columns)
    d_valid = xgb.DMatrix( data          = train_df.iloc[val_idx][features], 
                           label         = target.iloc[val_idx], 
                           feature_names = train_df.columns)

    watchlist = [(d_valid, 'valid')]
    model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, 
                      evals=watchlist,
                      early_stopping_rounds=early_stop, 
                      verbose_eval=verbose_eval, params=params)

    valid_pred = model.predict(xgb.DMatrix(train_df.iloc[val_idx][features], 
                                           feature_names=train_df.columns), 
                               ntree_limit=model.best_ntree_limit)
    


    test_pred = model.predict(xgb.DMatrix(test_df[features], 
                                          feature_names=train_df.columns), 
                              ntree_limit=model.best_ntree_limit)

    oof_xgb[val_idx]   = valid_pred
#     predictions   += test_pred



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_xgb)))

sub_df = pd.DataFrame({"id":ts_ids})
sub_df["target"] = test_pred.mean(axis=1)
# sub_df.target = np.where(sub_df.target>0.5,1,0)

sub_df.columns = sub.columns
sub_df.to_csv('submission/xgboost_{}.csv'.format(file_path), index=None)

sub_df.head()

xgb_imp = pd.DataFrame(data=[list(model.get_fscore().keys()), list(model.get_fscore().values())]).T
xgb_imp.columns = ['feature','imp']
xgb_imp = xgb_imp.sort_values(by='imp', ascending=False)
plt.figure(figsize=(12,15))
plt.barh(xgb_imp.feature, xgb_imp.imp)




from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import Pool, CatBoostClassifier
import catboost

features = train_df.columns
 
model = CatBoostClassifier(loss_function         = "Logloss", 
                           eval_metric           = "AUC",
                           random_strength       = 1.5,
                           border_count          = 128,
                           scale_pos_weight      = 3.507,
                           depth                 = 4, 
                           early_stopping_rounds = 50,
                           random_seed           = 1337,
                           task_type             = 'CPU', 
#                            subsample           = 0.7, 
                           iterations            = 10000, 
                           learning_rate         = 0.09
                          )
    

n_split = splits
kf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_seed)

y_valid_pred = 0 * target
oof_cat = np.zeros(len(train_df))
y_test_pred = 0


for idx, (train_index, valid_index) in enumerate(kf.split(train_df, target)):
    y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
    X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
    _train = Pool(X_train, label=y_train)#, cat_features=cate_features_index)
    _valid = Pool(X_valid, label=y_valid)#, cat_features=cate_features_index)
    print( "\nFold ", idx)

    fit_model = model.fit(_train, 
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=50
                         )
    pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  auc = ", roc_auc_score(y_valid, pred) )
    
    try:
        y_valid_pred.iloc[valid_index] = pred
    except:
        y_valid_pred.iloc[valid_index] = pred.reshape(-1,1)
    
    y_test_pred += fit_model.predict_proba(test_df)[:,1]

    oof_cat[valid_index] = fit_model.predict_proba(train_df.iloc[valid_index][features])[:,1]
     
    print("="*60)

y_test_pred /= n_split


print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_cat)))
sub_df = pd.DataFrame({"ID_code":test_id})
sub_df["target"] = y_test_pred
sub_df.columns = sub.columns

 
sub_df.to_csv('submission/catboost_{}.csv'.format(file_path), index=None)


cat_imp = pd.DataFrame(data=[fit_model.feature_names_, list(fit_model.feature_importances_)]).T
cat_imp.columns = ['feature','imp']
cat_imp = cat_imp.sort_values(by='imp', ascending=False)
plt.figure(figsize=(12,15))
plt.barh(cat_imp.feature, cat_imp.imp)








from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb

features = train_df.columns
param = {
    'bagging_freq'           : 5,
    'bagging_fraction'       : 0.33,
    'boost_from_average'     : 'false',
    'boost'                  : 'gbdt',
    'feature_fraction'       : 0.3,
    'learning_rate'          : 0.01,
    'max_depth'              : -1,
    'metric'                 : 'auc',
    'min_data_in_leaf'       : 100,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves'             : 30,
    'num_threads'            : 4,
    'tree_learner'           : 'serial',
    'objective'              : 'binary',
    'verbosity'              : 1,
#     'lambda_l1'              : 0.001,
    'lambda_l2'              : 0.5
}   

n_splits = splits
num_round = 10000
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
oof_lgb = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

temp = train_df.apply(lambda x: pd.Series.value_counts(x).shape[0])
cat_columns = list(temp[temp<50].index)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print(trn_idx.shape, val_idx.shape)
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx],
                          feature_name=list(train_df.columns))#,
#                           categorical_feature=cat_columns)

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx],
                          feature_name=list(train_df.columns))#,
#                           categorical_feature=cat_columns) 

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], 
                    verbose_eval=50, early_stopping_rounds = 25)
    oof_lgb[val_idx] = clf.predict(train_df.iloc[val_idx][features], 
                               num_iteration=clf.best_iteration)
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
    
    print( "  auc = ", roc_auc_score(target.iloc[val_idx], oof_lgb[val_idx]) )
    print("="*60)

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_lgb)))

sub_df = pd.DataFrame({"ID_code":test_id})
sub_df["target"] = predictions

sub_df.columns = sub.columns
sub_df.to_csv('submission/lightgbm_target_{}.csv'.format(file_path), index=None)



lgb_imp = pd.DataFrame(data=[clf.feature_name(), list(clf.feature_importance())]).T
lgb_imp.columns = ['feature','imp']
lgb_imp = lgb_imp.sort_values(by='imp', ascending=False)
plt.figure(figsize=(12,15))
plt.barh(lgb_imp.feature, lgb_imp.imp)















def run_cat(splits, file_path, train_df, target, test_df, test_id, sub, depth):
    
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from catboost import Pool, CatBoostClassifier
    import catboost

    features = train_df.columns
    random_seed = 2019
    
    model = CatBoostClassifier(loss_function         = "Logloss", 
                               eval_metric           = "AUC",
                               random_strength       = 1.5,
                               border_count          = 128,
                               scale_pos_weight      = 3.507,
                               depth                 = depth, 
                               early_stopping_rounds = 50,
                               random_seed           = 1337,
                               task_type             = 'CPU', 
    #                            subsample           = 0.7, 
                               iterations            = 10000, 
                               learning_rate         = 0.09,
                               thread_count          = 4
                              )


    n_split = splits
    kf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_seed)

    y_valid_pred = 0 * target
    oof_cat = np.zeros(len(train_df))
    y_test_pred = 0


    for idx, (train_index, valid_index) in enumerate(kf.split(train_df, target)):
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
        _train = Pool(X_train, label=y_train)#, cat_features=cate_features_index)
        _valid = Pool(X_valid, label=y_valid)#, cat_features=cate_features_index)
        print( "\nFold ", idx)

        fit_model = model.fit(_train, 
                              eval_set=_valid,
                              use_best_model=True,
                              verbose=100
                             )
        pred = fit_model.predict_proba(X_valid)[:,1]
        print( "  auc = ", roc_auc_score(y_valid, pred) )

        try:
            y_valid_pred.iloc[valid_index] = pred
        except:
            y_valid_pred.iloc[valid_index] = pred.reshape(-1,1)

        y_test_pred += fit_model.predict_proba(test_df)[:,1]

        oof_cat[valid_index] = fit_model.predict_proba(train_df.iloc[valid_index][features])[:,1]

        print("="*60)

    y_test_pred /= n_split


    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_cat)))
    sub_df = pd.DataFrame({"ID_code":test_id})
    sub_df["target"] = y_test_pred
    sub_df.columns = sub.columns


    sub_df.to_csv('submission/catboost_{}.csv'.format(file_path), index=None)


    cat_imp = pd.DataFrame(data=[fit_model.feature_names_, list(fit_model.feature_importances_)]).T
    cat_imp.columns = ['feature','imp']
    cat_imp = cat_imp.sort_values(by='imp', ascending=False).head(50)
    plt.figure(figsize=(12,15))
    plt.barh(cat_imp.feature, cat_imp.imp)
    plt.show()
    
    return fit_model, cat_imp










