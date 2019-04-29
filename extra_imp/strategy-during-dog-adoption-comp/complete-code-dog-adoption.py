


# cols =  ['Breed1', 'sentiment_magnitude', 'PhotoAmt', 'Age', 'Length_Description',
#          'Quantity', 'Breed2', 'State', 'Fee',#'sentiment_score'
#          'main_breed_BreedName', 'second_breed_BreedName', 'RescuerID_COUNT',
#          'Lengths_sentiment_entities', 'Length_metadata_annots_top_desc']
# len(cols)

# data = pd.get_dummies(temp_all[cols], columns=cols)
# print(data.shape)

# from keras.layers import Dense, Flatten, Input
# from keras import Model, regularizers

# inp = Input(shape=(1055,))
# x = Dense(200, activation='relu')(inp)
# x = Dense(70, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)
# x = Dense(200, activation='relu')(x)
# x = Dense(1055, activation='sigmoid')(x)
# model = Model(inp, x)
# # print(model.summary())


# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# early = EarlyStopping( monitor='val_loss',verbose=1, patience=3)
# reduce_lr = ReduceLROnPlateau( monitor='val_loss', factor=0.1,
#                                patience=1, verbose=1, mode='auto', 
#                                min_delta=0.0001, min_lr=0,)

# noise_factor = 0.5
# # data = data.values
# # data_tr = data[:14993, :]
# # data_ts = data[14993:, :]
# # data_tr_noisy = data_tr + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data_tr.shape) 
# # data_ts_noisy = data_ts + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data_ts.shape) 

# # data_tr_noisy = np.clip(data_tr_noisy, 0., 1.)
# # data_ts_noisy = np.clip(data_ts_noisy, 0., 1.)
# print(data_tr_noisy.shape, data_ts_noisy.shape)

# model.compile(optimizer='adadelta', loss='binary_crossentropy')
# history = model.fit(data_tr_noisy, data_tr, 
#                     epochs=100, batch_size=200, 
#                     validation_data=[data_ts_noisy, data_ts], 
#                     callbacks=[early, reduce_lr])

# model1 = Model(inp, model.layers[2].output)
# # print(model1.summary())
# preds = model1.predict(data, batch_size=2000)
# print(preds.shape)

# nn_feat = pd.DataFrame(data=preds, columns=['nn_'+str(i) for i in range(70)])
# save_noisy = pd.concat([temp_all[cols], nn_feat], axis=1)
# print("save_noisy: ", save_noisy.shape)

# cols_to_use = []
# for col,(i,j) in enumerate(zip(nn_feat.min(axis=0), nn_feat.max(axis=0))):
#     if (i+j) != 0.0:
#         cols_to_use.append(col)
# #         print(col, i, j)
# print("final decision: ", len(cols_to_use))

# save_noisy1 = pd.concat([temp_all[cols], nn_feat.iloc[:,cols_to_use]], axis=1)
# print("save_noisy1: ", save_noisy1.shape)



# better_noisy_deep = save_noisy1
# better_noisy_deep.shape

del save_noisy1, save_noisy, data, data_tr, data_tr_noisy, data_ts, data_ts_noisy, model, model1
gc.collect()
gc.collect()
gc.collect()






from keras.layers import Dense, Flatten, Input
from keras import Model, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

inp = Input(shape=(1055,))
x = Dense(50, activation='relu', activity_regularizer=regularizers.l1(10e-5))(inp)
x = Dense(1055, activation='sigmoid')(x)
model = Model(inp, x)
# print(model.summary())

early = EarlyStopping( monitor='val_loss',verbose=1, patience=3)
reduce_lr = ReduceLROnPlateau( monitor='val_loss', factor=0.1,
                               patience=1, verbose=1, mode='auto', 
                               min_delta=0.0001, min_lr=0,)
model.compile(optimizer='adadelta', loss='binary_crossentropy')
history = model.fit(data_tr_noisy, data_tr, 
                    epochs=100, batch_size=200, 
                    validation_data=[data_ts_noisy, data_ts], 
                    callbacks=[early, reduce_lr])

model1 = Model(inp, model.layers[1].output)
# print(model1.summary())
preds = model1.predict(data, batch_size=2000)
print(preds.shape)

nn_feat = pd.DataFrame(data=preds, columns=['nn_'+str(i) for i in range(50)])
save_noisy = pd.concat([temp_all[cols], nn_feat], axis=1)
# save_noisy.to_csv('noisy_50.csv', index=None)
print(save_noisy.shape)

cols_to_use = []
for col,(i,j) in enumerate(zip(nn_feat.min(axis=0), nn_feat.max(axis=0))):
    if (i+j) != 0.0:
        cols_to_use.append(col)
#         print(col, i, j)
        
print("final decision: ", len(cols_to_use))
save_noisy1 = pd.concat([temp_all[cols], nn_feat.iloc[:,cols_to_use]], axis=1)
print(save_noisy1.shape)
better_noisy_small = save_noisy1




















############################### TARGET ENCODING ###############################
# danger_col = ['Breed1', 'sentiment_magnitude', 'PhotoAmt', 'Age', 
#               'Quantity', 'sentiment_score', 'Breed2', 'State', 'Fee']
danger_col = ['Breed1', 'sentiment_magnitude', 'PhotoAmt', 'Age', 
         'Quantity', 'sentiment_score', 'Breed2', 'State', 'Fee',
         'main_breed_BreedName', 'second_breed_BreedName', 'RescuerID_COUNT', 'Length_Description',
         'Lengths_sentiment_entities', 'Length_metadata_annots_top_desc']
for col in danger_col:
    print(col, temp_all[col].value_counts().shape)


# # main_breed_BreedName == (176,)
# # second_breed_BreedName == (135,)
# # RescuerID_COUNT == (61,)
# # Length_Description == (1478,)
# # Length_metadata_annots_top_desc == (1615,)
# # Lengths_sentiment_entities == (538,)
# # VideoAmt == (9,)

# a= pd.qcut(temp_all['Lengths_sentiment_entities'],6, duplicates='drop', labels=[0,1,2,3,4,5])
# b = pd.qcut(temp_all['Length_metadata_annots_top_desc'],8, duplicates='drop', labels=[0,1,2,3,4,5,6,7])
# c = pd.qcut(temp_all['Length_Description'],7, duplicates='drop', labels=[0,1,2,3,4,5,6])
# d = pd.qcut(temp_all['RescuerID_COUNT'],5, duplicates='drop', labels=[0,1,2,3])
# e = pd.qcut(temp_all['main_breed_BreedName'],5, duplicates='drop', labels=[0,1,2,3])
# f = pd.qcut(temp_all['second_breed_BreedName'],6, duplicates='drop', labels=[0,1])

# col_more_to_use = [  'main_breed_BreedName', 'second_breed_BreedName', 'RescuerID_COUNT', 'Length_Description',\
#                      'Lengths_sentiment_entities', 'Length_metadata_annots_top_desc']
# print(temp_all.shape)
# # danger_col = ['Breed1', 'sentiment_magnitude', 'PhotoAmt', 'Age', 
# #               'Quantity', 'sentiment_score', 'Breed2', 'State', 'Fee']
# col_more_to_use_value = [a,b,c,d,e,f]
# for col, value in zip(col_more_to_use, col_more_to_use_value):
#     temp_all[col+'_qcut1'] = value
# print(temp_all.shape)


# cols2 = [col+'_qcut1' for col in col_more_to_use]

# temp_all[cols2] = temp_all[cols2].astype(int)
# print("At starting: ",temp_all.shape)
# import category_encoders as ce
# for col in cols2:
#     try:
#         bde = ce.PolynomialEncoder(cols=[col])
#         # BackwardDifferenceEncoder(cols=['CHAS', 'RAD']).fit(X, y)
#         bde.fit(temp_all[[col]][:14993], X_train_non_null['AdoptionSpeed'])
#         encoded = bde.transform(temp_all[[col]])
#         encoded.columns = [col+'_pe1' for col in encoded.columns]
#         temp_all = pd.concat([temp_all,encoded.iloc[:,1:]],axis=1)
#         add_cols.append(encoded.columns[1:])
#         print(col)
#     except:
#         print("not encoded: ", col)
#         pass
# #     temp_all = pd.concat([temp_all,encoded[encoded.columns[1:]]], axis=1)
# print("After pe encoding: ", temp_all.shape)
# print(" ========= ")
# # add_cols = []
# import category_encoders as ce
# for col in cols2:
# #     print('*'*40)
#     try:
#         bde = ce.BackwardDifferenceEncoder(cols=[col])
#         # BackwardDifferenceEncoder(cols=['CHAS', 'RAD']).fit(X, y)
#         bde.fit(temp_all[[col]][:14993], X_train_non_null['AdoptionSpeed'])
#         encoded = bde.transform(temp_all[[col]])
#         encoded.columns = [col+'_bd1' for col in encoded.columns]
#         temp_all = pd.concat([temp_all,encoded.iloc[:,1:]],axis=1)
#         add_cols.append(encoded.columns[1:])
#         print(col)
#     except:
#         pass
# #     temp_all = pd.concat([temp_all,encoded[encoded.columns[1:]]], axis=1)
# print("After BE Encoding: ", temp_all.shape)
















###################### CATBOOST #####################
# from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.metrics import roc_auc_score
# from catboost import Pool, CatBoostClassifier
# import catboost


# model = CatBoostClassifier(   iterations=999999,
#                               max_depth=2,
#                               learning_rate=0.02,
#                               colsample_bylevel=0.03,
#                               objective="Logloss",
#                               eval_metric="AUC")
                                  
# # model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC")#, iterations=300)
# # feat_imp = pd.DataFrame()
# n_split = 5#10
# cur_model = 'meta_cat1_'
# # kf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=31415)
# kf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=44000)

# # kf = KFold(n_splits=n_split, random_state=42, shuffle=True)

# y_valid_pred = 0 * target
# # y_train_pred = 0 * target
# oof = np.zeros(len(combine_all_tr))

# y_test_pred = 0

# for idx, (train_index, valid_index) in enumerate(kf.split(combine_all_tr, target)):
#     y_train, y_valid = target[train_index], target[valid_index]
#     X_train, X_valid = combine_all_tr.iloc[train_index,:], combine_all_tr.iloc[valid_index,:]
#     _train = Pool(X_train, label=y_train)
#     _valid = Pool(X_valid, label=y_valid)
#     print( "\nFold ", idx)
    
#     fit_model = model.fit(_train, 
# #                           cat_features=cat_feature_indices,
#                           eval_set=_valid,
#                           use_best_model=True,
#                           verbose=200,
#                           early_stopping_rounds = 250
#                          )
#     pred = fit_model.predict_proba(X_valid)[:,1]
#     print( "  auc = ", roc_auc_score(y_valid, pred) )
#     y_valid_pred.iloc[valid_index] = pred
#     y_test_pred += fit_model.predict_proba(combine_all_ts)[:,1]

#     oof[valid_index] = fit_model.predict_proba(combine_all_tr.iloc[valid_index][features])[:,1]
     
#     fit_model.save_model('stacked_models/'+cur_model+str(idx)+'.txt')
# #     feat_imp[str(idx)+'_features'] = fit_model.feature_names_
# #     feat_imp[str(idx)+'_importance'] = fit_model.get_feature_importance()
# y_test_pred /= n_split


# # catboost.load_model(model_path)
# print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

# sub_df = pd.DataFrame({"ID_code":test_id})
# sub_df["target"] = y_test_pred
# sub_df.to_csv("stacked_models/test_"+cur_model+".csv", index=False)

# sub_df = pd.DataFrame({"ID_code":train_id})
# sub_df["target"] = oof
# # sub_df.to_csv("stacked_models/train_"+cur_model+".csv", index=False)

# # del train, test
# # gc.collect()
# # print("deleetd data train and test prepared for catboost...")






















######################## Hyperoptimization XGBoost #####################################

# def run_xgb(param):#, X_train, X_test, smoothing, noise_level=0.1):
#     min_samples_leaf        = int(param['min_samples_leaf'])
#     smoothing               = int(param['smoothing'])
#     noise_level             = param['noise_level']
    
#     X_train = X_train_non_null1
#     X_test = X_test_non_null1
#     n_splits = 2
#     verbose_eval = 1000
#     num_rounds = 60000
#     early_stop = 500

#     kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
# # oof_train_lgb1
#     oof_train = np.zeros((X_train.shape[0]))
#     oof_test = np.zeros((X_test.shape[0], n_splits))

#     i = 0
#     qwks = []
#     for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):
#         print("fold ",i)
#         X_tr = X_train.iloc[train_idx, :]
#         X_val = X_train.iloc[valid_idx, :]

#         y_tr = X_tr['AdoptionSpeed']
#         X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

#         y_val = X_val['AdoptionSpeed'].values
#         X_val = X_val.drop(['AdoptionSpeed'], axis=1)

#         cols = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
#                'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
#                'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'VideoAmt',
#                'PhotoAmt', 'main_breed_BreedName', 'second_breed_BreedName', 
#                 'RescuerID_COUNT']

#         for f in cols:
#             X_tr[f + "_te"], X_val[f + "_te"], X_test[f + "_te"] = target_encode(
#                                                             trn_series=X_tr[f],
#                                                             val_series=X_val[f],
#                                                             tst_series=X_test[f],
#                                                             target=y_tr,
#                                                             min_samples_leaf=int(min_samples_leaf),
#                                                             smoothing=int(smoothing),
#                                                             noise_level=noise_level
#                                                             )

#         y_tr = y_tr.values
#         d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
#         d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

#         watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#         xgb_params = {
#             'eval_metric': 'rmse',
#             'seed': 1337,
#             'eta': 0.0123,
#             'subsample': 0.8,
#             'colsample_bytree': 0.85,
#             'tree_method': 'gpu_hist',
#             'device': 'gpu',
#             'silent': 1,
#         }
#         model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
#                          early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=xgb_params)

#         valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
# #         temp = model.predict_proba(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
#         ############################
#         optR = OptimizedRounder()
#         optR.fit(valid_pred, y_val)
#         coefficients = optR.coefficients()
#         pred_val_y_k = optR.predict(valid_pred, coefficients)
#         qwk = quadratic_weighted_kappa(y_val, pred_val_y_k)
# #         qwk_scores.append(qwk)
#         print("QWK = ", qwk)
#     #########################
#         test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

#         oof_train[valid_idx] = valid_pred
#         oof_test[:, i] = test_pred
#         qwks.append(qwk)
#         i += 1
#     print("final qwk: ", np.mean(qwks))
#     return -np.mean(qwks)#model, oof_train, oof_test





# from skopt import BayesSearchCV
# from hyperopt import hp, tpe, fmin

# space = {
#     'min_samples_leaf': hp.uniform('min_samples_leaf', 30, 100), 
#     'smoothing':        hp.uniform('smoothing', 1, 50),  
#     'noise_level':      hp.uniform('noise_level', 0.0, 1.0),
# }
# best = fmin(fn=run_xgb,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=10)













############ LEAVE-ONE-OUT ################
# danger_col = ['Breed1', 'sentiment_magnitude', 'PhotoAmt', 'Age', 
#               'Quantity', 'sentiment_score', 'Breed2', 'State', 'Fee']
# a= pd.qcut(temp_all['Breed1'],5, duplicates='drop', labels=[0,1,2,3])
# b = pd.qcut(temp_all['sentiment_magnitude'],8, duplicates='drop', labels=[0,1,2,3,4,5,6,7])
# c = pd.qcut(temp_all['PhotoAmt'],5, duplicates='drop', labels=[0,1,2,3,4])
# d = pd.qcut(temp_all['Age'],5, duplicates='drop', labels=[0,1,2,3,4])
# e = pd.qcut(temp_all['Quantity'],5, duplicates='drop', labels=[0,1])
# f = pd.qcut(temp_all['sentiment_score'],7, duplicates='drop', labels=[0,1,2,3,4,5,6])
# g = pd.qcut(temp_all['Breed2'],5, duplicates='drop', labels=[0,1])
# h = pd.qcut(temp_all['State'],5, duplicates='drop', labels=[0,1,2,3])
# # i = pd.qcut(temp_all['sentiment_score'], 8, duplicates='drop', labels=[0, 1, 2, 3, 4, 5, 6, 7])
# fee_enc = [0 if ii is 0 else 1 for ii in temp_all.Fee ]


# a= pd.qcut(temp_all['Lengths_sentiment_entities'],6, duplicates='drop', labels=[0,1,2,3,4,5])
# b = pd.qcut(temp_all['Length_metadata_annots_top_desc'],8, duplicates='drop', labels=[0,1,2,3,4,5,6,7])
# c = pd.qcut(temp_all['Length_Description'],7, duplicates='drop', labels=[0,1,2,3,4,5,6])
# d = pd.qcut(temp_all['RescuerID_COUNT'],5, duplicates='drop', labels=[0,1,2,3])
# e = pd.qcut(temp_all['main_breed_BreedName'],5, duplicates='drop', labels=[0,1,2,3])
# f = pd.qcut(temp_all['second_breed_BreedName'],6, duplicates='drop', labels=[0,1])

# col_more_to_use = [  'main_breed_BreedName', 'second_breed_BreedName', 'RescuerID_COUNT', 'Length_Description',\
#                      'Lengths_sentiment_entities', 'Length_metadata_annots_top_desc']
# print(temp_all.shape)
# # danger_col = ['Breed1', 'sentiment_magnitude', 'PhotoAmt', 'Age', 
# #               'Quantity', 'sentiment_score', 'Breed2', 'State', 'Fee']
# col_more_to_use_value = [a,b,c,d,e,f]
# for col, value in zip(col_more_to_use, col_more_to_use_value):
#     temp_all[col+'_qcut1'] = value
# print(temp_all.shape)


# cols2 = [col+'_qcut1' for col in col_more_to_use]
cols2 = danger_col + col_more_to_use
temp_all[cols2] = temp_all[cols2].astype(int)
print("At starting: ",temp_all.shape)
import category_encoders as ce
for col in cols2:
    try:
        bde = ce.leave_one_out.LeaveOneOutEncoder(cols=[col])
#         bde = ce.PolynomialEncoder(cols=[col])
        # BackwardDifferenceEncoder(cols=['CHAS', 'RAD']).fit(X, y)
        bde.fit(temp_all[[col]][:14993], X_train_non_null['AdoptionSpeed'])
        encoded = bde.transform(temp_all[[col]])
        encoded.columns = [col+'_oo' for col in encoded.columns]
        temp_all = pd.concat([temp_all,encoded],axis=1)
        add_cols.append(encoded.columns[1:])
        print(col)
    except:
        print("not encoded: ", col)
        pass
#     temp_all = pd.concat([temp_all,encoded[encoded.columns[1:]]], axis=1)
print("After oo encoding: ", temp_all.shape)












##################### XGB with target encoding ###############################
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

xgb_params = {  'eval_metric': 'rmse', 'seed': 1337, 'eta': 0.0123, 'subsample': 0.8,
                'colsample_bytree': 0.85, 'tree_method': 'gpu_hist', 'device': 'gpu', 
                'silent': 1, 'max_depth':6}#, 'booster': 'dart'

def run_xgb(params, X_train, X_test):
    n_splits = 10
    verbose_eval = 1000
    num_rounds = 60000
    early_stop = 500

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    qwk_scores = []
    i = 0

    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):
        print("fold ",i)
        X_tr, X_val = X_train.iloc[train_idx, :], X_train.iloc[valid_idx, :]
        y_tr, y_val = X_tr['AdoptionSpeed'], X_val['AdoptionSpeed']
        X_tr, X_val = X_tr.drop(['AdoptionSpeed'], axis=1), X_val.drop(['AdoptionSpeed'], axis=1)

#         cols = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
#                'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
#                'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'VideoAmt',
#                'PhotoAmt', 'main_breed_BreedName', 'second_breed_BreedName', 
#                 'RescuerID_COUNT']
#         import category_encoders as ce
#         for col in cols:
#             bde = ce.target_encoder.TargetEncoder(cols=[col], min_samples_leaf=40, smoothing=1)
#             # BackwardDifferenceEncoder(cols=['CHAS', 'RAD']).fit(X, y)
#             bde.fit(X_tr[[col]], y_tr)
#             encoded = bde.transform(X_tr[[col]])
#             encoded.columns = [col+'_te' for col in encoded.columns]
#             X_tr = pd.concat([X_tr,encoded],axis=1)
            
#             encoded = bde.transform(X_val[[col]])
#             encoded.columns = [col+'_te' for col in encoded.columns]
#             X_val = pd.concat([X_val,encoded],axis=1)
#             encoded = bde.transform(X_test[[col]])
#             encoded.columns = [col+'_te' for col in encoded.columns]
#             X_test = pd.concat([X_test,encoded],axis=1)
            
#         for f in cols:
#             X_tr[f + "_avg"], X_val[f + "_avg"], X_test[f + "_avg"] = target_encode(
#                                                                     trn_series=X_tr[f], val_series=X_val[f],
#                                                                     tst_series=X_test[f], target=y_tr,
#                                                                     min_samples_leaf=50,smoothing=5,
#                                                                     noise_level=0.1)
#             target_encode(trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0)

        y_tr, y_val = y_tr.values, y_val.values
        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
        ############################
        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_val_y_k = optR.predict(valid_pred, coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_val_y_k)
        qwk_scores.append(qwk)
        print("QWK = ", qwk)
        #########################
        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)
        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred
        i += 1
    print("Avg qwk: ", np.mean(qwk_scores))
    return model, oof_train, oof_test






















######################