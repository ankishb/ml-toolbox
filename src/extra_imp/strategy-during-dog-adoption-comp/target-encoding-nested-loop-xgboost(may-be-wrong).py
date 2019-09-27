import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

xgb_params = {  'eval_metric': 'rmse', 'seed': 1337, 'eta': 0.0123, 'subsample': 0.8,
                'colsample_bytree': 0.85, 'tree_method': 'gpu_hist', 'device': 'gpu', 
                'silent': 1, 'max_depth':6}#, 'booster': 'dart'

def run_xgb_encoding(params, X_train, X_test):
    
    verbose_eval = 1000
    num_rounds = 60000
    early_stop = 500

    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    qwk_scores = []
    i = 0

    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):
        print("fold ",i)
        X_tst = X_test
        X_tr, X_val = X_train.iloc[train_idx, :], X_train.iloc[valid_idx, :]
        y_tr, y_val = X_tr['AdoptionSpeed'], X_val['AdoptionSpeed']
        X_tr, X_val = X_tr.drop(['AdoptionSpeed'], axis=1), X_val.drop(['AdoptionSpeed'], axis=1)

        ##################### NESTED CV ################
        n_splits_in = 5
        kf_in = StratifiedKFold(n_splits=n_splits_in, shuffle=True, random_state=1337)
        oof_train_in = np.zeros((X_train.shape[0]))
        oof_test_in = np.zeros((X_tst.shape[0], n_splits_in))
        oof_val_out = np.zeros((len(valid_idx), n_splits_in))
        
        qwk_scores_in = []
        i_in = 0
        import category_encoders as ce
        for train_idx_in, valid_idx_in in kf.split(X_tr, y_tr):
            print("Inner fold ",i_in)
            print(train_idx_in.shape, valid_idx_in.shape, X_tr.shape, X_val.shape)
            X_val_out = X_val
            X_tr_in, X_val_in = X_tr.iloc[train_idx_in, :], X_tr.iloc[valid_idx_in, :]
            y_tr_in, y_val_in = y_tr.iloc[train_idx_in, :], y_tr.iloc[valid_idx_in, :]
#             X_tr_in, X_val_in = X_tr.drop(['AdoptionSpeed'], axis=1), X_val.drop(['AdoptionSpeed'], axis=1)
            
            cols = ['Age', 'Breed1', 'Breed2','Quantity', 'Fee', 'State', 'VideoAmt','PhotoAmt']#, 
#                     'main_breed_BreedName', 'second_breed_BreedName','RescuerID_COUNT']
            for f in cols:
                bde = ce.target_encoder.TargetEncoder(cols=[f], min_samples_leaf=50, smoothing=10)
#                 bde = ce.PolynomialEncoder(cols=[col])
                # BackwardDifferenceEncoder(cols=['CHAS', 'RAD']).fit(X, y)
                bde.fit(X_tr_in[[col]], y_tr_in)
                encoded = bde.transform(X_tr_in[[col]])
                encoded.columns = [col+'_te' for col in encoded.columns]
                X_tr_in = pd.concat([X_tr_in, encoded.iloc[:,:]],axis=1)
                
                encoded = bde.transform(X_val_in[[col]])
                encoded.columns = [col+'_te' for col in encoded.columns]
                X_val_in = pd.concat([X_val_in,encoded.iloc[:,:]],axis=1)
                
                encoded = bde.transform(X_tst[[col]])
                encoded.columns = [col+'_te' for col in encoded.columns]
                X_tst = pd.concat([X_tst,encoded.iloc[:,:]],axis=1)   
                
                encoded = bde.transform(X_val_out[[col]])
                encoded.columns = [col+'_te' for col in encoded.columns]
                X_val_out = pd.concat([X_val_out,encoded.iloc[:,:]],axis=1)
                
#                 X_tr_in[f + "_avg"], X_val_in[f + "_avg"], X_tst[f + "_avg"] = target_encode(
#                                                                         trn_series=X_tr_in[f], val_series=X_val_in[f],
#                                                                         tst_series=X_tst[f], target=y_tr_in,
#                                                                         min_samples_leaf=50,smoothing=5,
#                                                                         noise_level=0.1)
    #             target_encode(trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0)

            y_tr_in, y_val_in = y_tr_in.values, y_val_in.values
            d_train = xgb.DMatrix(data=X_tr_in, label=y_tr_in, feature_names=X_tr_in.columns)
            d_valid = xgb.DMatrix(data=X_val_in, label=y_val_in, feature_names=X_val_in.columns)

            watchlist = [(d_train, 'train'), (d_valid, 'valid')]
            model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                             early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)
            ############################ Inner Loop ############################
            valid_pred_in = model.predict(xgb.DMatrix(X_val_in, feature_names=X_val_in.columns), ntree_limit=model.best_ntree_limit)
            optR = OptimizedRounder()
            optR.fit(valid_pred_in, y_val_in)
            coefficients = optR.coefficients()
            pred_val_y_k = optR.predict(valid_pred_in, coefficients)
            qwk = quadratic_weighted_kappa(y_val_in, pred_val_y_k)
            qwk_scores_in.append(qwk)
            print("Inner QWK = ", qwk)
            oof_train_in[valid_idx_in] = valid_pred_in
            #########################        ############################
            
            ############################ Outer Loop ############################
            valid_pred_out = model.predict(xgb.DMatrix(X_val_out, feature_names=X_val_out.columns), ntree_limit=model.best_ntree_limit)
            optR = OptimizedRounder()
            optR.fit(valid_pred_in, y_val_in)
            coefficients = optR.coefficients()
            pred_val_y_k = optR.predict(valid_pred_in, coefficients)
            qwk = quadratic_weighted_kappa(y_val_in, pred_val_y_k)
#             qwk_scores_in.append(qwk)
            print("Outer QWK = ", qwk)
            oof_val_out[:, i_in] = valid_pred_out
            #########################        ############################
            test_pred = model.predict(xgb.DMatrix(X_tst, feature_names=X_tst.columns), ntree_limit=model.best_ntree_limit)
            oof_test_in[:, i_in] = test_pred
            i_in += 1
        print("Avg qwk: {}, for fold: {} ".format(np.mean(qwk_scores_in), i))
        qwk_scores.append(np.mean(qwk_scores_in))
        print("check shapes of the inner loop preds: ", oof_train_in.shape, oof_test_in.shape)
        oof_test[:, i] = np.mean(oof_test_in, axis=1)
        oof_train[valid_idx] = np.mean(oof_val_out, axis=1)
        ############################ NESTED CV ########################
#         cols = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
#                'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
#                'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'VideoAmt',
#                'PhotoAmt', 'main_breed_BreedName', 'second_breed_BreedName', 
#                 'RescuerID_COUNT']
#         for f in cols:
#             X_tr[f + "_avg"], X_val[f + "_avg"], X_tst[f + "_avg"] = target_encode(
#                                                                     trn_series=X_tr[f], val_series=X_val[f],
#                                                                     tst_series=X_tst[f], target=y_tr,
#                                                                     min_samples_leaf=50,smoothing=5,
#                                                                     noise_level=0.1)
#             target_encode(trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0)

#         y_tr, y_val = y_tr.values, y_val.values
#         d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
#         d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

#         watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#         model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
#                          early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

#         valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
#         ############################
#         optR = OptimizedRounder()
#         optR.fit(valid_pred, y_val)
#         coefficients = optR.coefficients()
#         pred_val_y_k = optR.predict(valid_pred, coefficients)
#         qwk = quadratic_weighted_kappa(y_val, pred_val_y_k)
#         qwk_scores.append(qwk)
#         print("QWK = ", qwk)
#         #########################
#         test_pred = model.predict(xgb.DMatrix(X_tst, feature_names=X_tst.columns), ntree_limit=model.best_ntree_limit)
#         oof_train[valid_idx] = valid_pred
#         oof_test[:, i] = test_pred
#         i += 1
#     print("Avg qwk: ", np.mean(qwk_scores))
    return model, oof_train, oof_test
















cols = ['Age', 'Breed1', 'Breed2','Quantity', 'Fee', 'State', 'VideoAmt','PhotoAmt']
import category_encoders as ce
# for col in cols:
col = cols[1]

bde = ce.target_encoder.TargetEncoder(cols=[col], min_samples_leaf=50, smoothing=10)
#                 bde = ce.PolynomialEncoder(cols=[col])
# BackwardDifferenceEncoder(cols=['CHAS', 'RAD']).fit(X, y)
bde.fit(df1[[col]], df1.AdoptionSpeed)
bde.transform(df1[[col]])