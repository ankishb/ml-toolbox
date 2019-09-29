def softmax(score):
    score = np.asarray(score, dtype=float)
    score = np.exp(score-np.max(score))
    score /= np.sum(score, axis=1)[:,np.newaxis]
    return score

## soft version of kappa score using the class probability
## inspired by @George Mohler in the Higgs competition
## https://www.kaggle.com/c/higgs-boson/forums/t/10286/customize-loss-function-in-xgboost/53459#post53459
## NOTE: As also discussed in the above link, it is hard to tune the hessian to get it to work.
def softkappaobj(preds, dtrain):
    ## label are in [0,1,2,3] as required by XGBoost for multi-classification
    labels = dtrain.get_label() + 1
    labels = np.asarray(labels, dtype=int)
    preds = softmax(preds)
    M = preds.shape[0]
    N = preds.shape[1]

    ## compute O (enumerator)
    O = 0.0
    for j in range(N):
        wj = (labels - (j+1.))**2
        O += np.sum(wj * preds[:,j])
    
    ## compute E (denominator)
    hist_label = np.bincount(labels)[1:]
    hist_pred = np.sum(preds, axis=0)
    E = 0.0
    for i in range(N):
        for j in range(N):
            E += pow(i - j, 2.0) * hist_label[i] * hist_pred[j]

    ## compute gradient and hessian
    grad = np.zeros((M, N))
    hess = np.zeros((M, N))
    for n in range(N):
        ## first-order derivative: dO / dy_mn
        dO = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            dO += ((labels - (j+1.))**2) * preds[:,n] * (indicator - preds[:,j])
        ## first-order derivative: dE / dy_mn
        dE = np.zeros((M))
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                dE += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (indicator - preds[:,k])
        ## the grad
        grad[:,n] = -M * (dO * E - O * dE) / (E**2)
        
        ## second-order derivative: d^2O / d (y_mn)^2
        d2O = np.zeros((M))
        for j in range(N):
            indicator = float(n == j)
            d2O += ((labels - (j+1.))**2) * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,j])
       
        ## second-order derivative: d^2E / d (y_mn)^2
        d2E = np.zeros((M))
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                d2E += pow(k-l, 2.0) * hist_label[l] * preds[:,n] * (1 - 2.*preds[:,n]) * (indicator - preds[:,k])
        ## the hess
        hess[:,n] = -M * ((d2O * E - O * d2E)*(E**2) - (dO * E - O * dE) * 2. * E * dE) / (E**4)

    grad *= -1.
    hess *= -1.
    # this pure hess doesn't work in my case, but the following works ok
    # use a const
    #hess = 0.000125 * np.ones(grad.shape, dtype=float)
    # or use the following...
    scale = 0.000125 / np.mean(abs(hess))
    hess *= scale
    hess = np.abs(hess) # It works!! no idea...
    grad.shape = (M*N)
    hess.shape = (M*N)
    return grad, hess

# evalerror is your customized evaluation function to 
# 1) decode the class probability 
# 2) compute quadratic weighted kappa
def evalerror(preds, dtrain):
    ## label are in [0,1,2,3] as required by XGBoost for multi-classification
    labels = dtrain.get_label() + 1
    ## class probability
    preds = softmax(preds)
    ## decoding (naive argmax decoding)
    pred_labels = np.argmax(preds, axis=1) + 1
    ## compute quadratic weighted kappa (using implementation from @Ben Hamner
    ## https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py
    kappa = quadratic_weighted_kappa(labels, pred_labels)
    return 'kappa', kappa











import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

xgb_class_params = {
#     'eval_metric'       : 'rmse',
    'seed'              : 1337,
    'eta'               : 0.0083,#0.0123,
    'subsample'         : 0.9288,#0.8,
    'colsample_bytree'  : 0.696,#0.85,
    'tree_method'       : 'gpu_hist',
    'device'            : 'gpu',
    'silent'            : 1,
    'alpha'             : 0.7129,#1,
    'lambda'            : 1.8005,#1,
    'max_depth'         : 4,
    'gamma'             : 49.806,
    'booster'           : 'gblinear',
    'num_class'         : 5
}




def run_xgb_class(params, X_train, X_test):
    
#     X_train, X_test = X_train_, X_test_
    n_splits = 5
    verbose_eval = 1000
    num_rounds = 60000
    early_stop = 500

    feature_importance_df = pd.DataFrame()
    from sklearn.model_selection import StratifiedKFold, GroupKFold
#     kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
    kf = GroupKFold(n_splits=n_splits)
    
    oof_train = np.zeros((X_train.shape[0], 5))
    oof_test = np.zeros((X_test.shape[0], 5, n_splits))

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
        
#         bst = xgb.train(param, dtrain, num_round, watchlist, obj=softkappaobj, feval=evalerror)

#         ## make prediction (class probability)
#         pred = softmax(bst.predict(dvalid))

        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params,
                         obj=softkappaobj, feval=evalerror)

        valid_pred = model.predict_proba(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)        
        test_pred = model.predict_proba(xgb.DMatrix(X_test, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)

        oof_train[valid_idx,:] = valid_pred
        oof_test[:, :, i] = test_pred
        
        fold_importance_df = pd.DataFrame.from_dict(model.get_fscore(), 
                                                    orient='index',
                                                    columns=['feat']).reset_index()
        fold_importance_df.columns = ['feature','importance']
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        i += 1

    return model, oof_train, oof_test, feature_importance_df














