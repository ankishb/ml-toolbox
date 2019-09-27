# from keras.layers import Dense, Flatten, Input, Dropout
# from keras import Model, regularizers
# from keras.optimizers import Adam
# def get_model1(inp_size):
#     # Flag_for9, Flag_for6
#     inp = Input(shape=(inp_size,))
#     x = Dense(100, activation='relu')(inp)
#     x = Dropout(0.25)(x)
# #     x = Dense(70, activation='relu')(x)#, activity_regularizer=regularizers.l1(10e-5))(x)
# #     x = Dropout(0.25)(x)
#     x = Dense(100, activation='relu')(x)
#     x = Dropout(0.25)(x)
#     x = Dense(1, activation='linear')(x)
#     model = Model(inp, x)
#     opt = Adam()
#     model.compile(optimizer=opt, loss='mse',)
#     return model

# def get_model2(inp_size):
#     inp = Input(shape=(inp_size,))
#     x = Dense(100, activation='relu')(inp)
#     x = Dropout(0.25)(x)
# #     x = Dense(100, activation='relu')(x)#, activity_regularizer=regularizers.l1(10e-5))(x)
# #     x = Dropout(0.25)(x)
#     x = Dense(100, activation='relu')(x)
#     x = Dropout(0.25)(x)
#     x = Dense(1, activation='linear')(x)
#     model = Model(inp, x)
#     opt = Adam()
#     model.compile(optimizer=opt, loss='mse',)
#     return model
# # print(model.summary())


# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# early = EarlyStopping( monitor='val_loss',verbose=1, patience=3)
# reduce_lr = ReduceLROnPlateau( monitor='val_loss', factor=0.2,
#                                patience=2, verbose=1, mode='auto', 
#                                min_delta=0.0001, min_lr=0,)











# from keras.callbacks import *

# class CyclicLR(Callback):
#     def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
#                  gamma=1., scale_fn=None, scale_mode='cycle'):
#         super(CyclicLR, self).__init__()

#         self.base_lr = base_lr
#         self.max_lr = max_lr
#         self.step_size = step_size
#         self.mode = mode
#         self.gamma = gamma
#         if scale_fn == None:
#             if self.mode == 'triangular':
#                 self.scale_fn = lambda x: 1.
#                 self.scale_mode = 'cycle'
#             elif self.mode == 'triangular2':
#                 self.scale_fn = lambda x: 1/(2.**(x-1))
#                 self.scale_mode = 'cycle'
#             elif self.mode == 'exp_range':
#                 self.scale_fn = lambda x: gamma**(x)
#                 self.scale_mode = 'iterations'
#         else:
#             self.scale_fn = scale_fn
#             self.scale_mode = scale_mode
#         self.clr_iterations = 0.
#         self.trn_iterations = 0.
#         self.history = {}

#         self._reset()

#     def _reset(self, new_base_lr=None, new_max_lr=None,
#                new_step_size=None):
#         """Resets cycle iterations.
#         Optional boundary/step size adjustment.
#         """
#         if new_base_lr != None:
#             self.base_lr = new_base_lr
#         if new_max_lr != None:
#             self.max_lr = new_max_lr
#         if new_step_size != None:
#             self.step_size = new_step_size
#         self.clr_iterations = 0.
        
#     def clr(self):
#         cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
#         x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
#         if self.scale_mode == 'cycle':
#             return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
#         else:
#             return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
#     def on_train_begin(self, logs={}):
#         logs = logs or {}

#         if self.clr_iterations == 0:
#             K.set_value(self.model.optimizer.lr, self.base_lr)
#         else:
#             K.set_value(self.model.optimizer.lr, self.clr())        
            
#     def on_batch_end(self, epoch, logs=None):
        
#         logs = logs or {}
#         self.trn_iterations += 1
#         self.clr_iterations += 1

#         self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
#         self.history.setdefault('iterations', []).append(self.trn_iterations)

#         for k, v in logs.items():
#             self.history.setdefault(k, []).append(v)
        
#         K.set_value(self.model.optimizer.lr, self.clr())
    
# class QWKEvaluation(Callback):
#     def __init__(self, validation_data=(), interval=1):
#         super(Callback, self).__init__()

#         self.interval = interval
#         self.history = []
#         self.X_val, self.y_val = validation_data
        
#     def on_epoch_end(self, epoch, logs={}):
#         if epoch % self.interval == 0:
#             y_pred = self.model.predict(self.X_val, batch_size=1000, verbose=0)
#             y_pred = eval_predict(self.y_val, y_pred)
            
#             score = quadratic_weighted_kappa(list(self.y_val.reshape(-1,)), list(y_pred))
#             print("QWK - epoch: %d - score: %.6f \n" % (epoch+1, score))
#             self.history.append(score)
#             if score >= max(self.history): self.model.save('checkpoint.h5')

# def eval_predict(y=[], y_pred=[], coeffs=None, ret_coeffs=False):
#     optR = OptimizedRounder()
#     if not coeffs:
#         optR.fit(y_pred.reshape(-1,), y.reshape(-1,))
#         coeffs = optR.coefficients()
#     if ret_coeffs: return optR.coefficients()
#     return optR.predict(y_pred.reshape(-1,), coeffs)





# Flag_for9 = 0
# Flag_for6 = 0
# try:
#     train_stack = np.vstack([oof_train_xgb1, oof_train_lgb1, oof_train_cat1, \
#                              oof_train_xgb2, oof_train_lgb2, oof_train_cat2, \
#                              oof_train_xgb3, oof_train_lgb3, oof_train_cat3]).transpose()
#     test_stack = np.vstack([np.mean(oof_test_xgb1, axis=1), np.mean(oof_test_lgb1, axis=1), np.mean(oof_test_cat1, axis=1),\
#                            np.mean(oof_test_xgb2, axis=1), np.mean(oof_test_lgb2, axis=1), np.mean(oof_test_cat2, axis=1),\
#                            np.mean(oof_test_xgb3, axis=1), np.mean(oof_test_lgb3, axis=1), np.mean(oof_test_cat3, axis=1)]).transpose()
#     train_stack = pd.DataFrame(data=train_stack, columns=['xgb1','lgb1','cat1','xgb2','lgb2','cat2','xgb3','lgb3','cat3'])
#     test_stack  = pd.DataFrame(data=test_stack,  columns=['xgb1','lgb1','cat1','xgb2','lgb2','cat2','xgb3','lgb3','cat3'])
#     print("used All model: ")
#     Flag_for9 = 1
# except:
#     train_stack = np.vstack([oof_train_xgb1, oof_train_lgb1, oof_train_cat1, \
# #                              oof_train_xgb2, oof_train_lgb2, \
#                              oof_train_xgb3, oof_train_lgb3, oof_train_cat3]).transpose()
#     test_stack = np.vstack([np.mean(oof_test_xgb1, axis=1), np.mean(oof_test_lgb1, axis=1), np.mean(oof_test_cat1, axis=1),\
# #                            np.mean(oof_test_xgb2, axis=1), np.mean(oof_test_lgb2, axis=1),\
#                            np.mean(oof_test_xgb3, axis=1), np.mean(oof_test_lgb3, axis=1), np.mean(oof_test_cat3, axis=1)]).transpose()
#     train_stack = pd.DataFrame(data=train_stack, columns=['xgb1','lgb1','cat1','xgb3','lgb3','cat3'])
#     test_stack  = pd.DataFrame(data=test_stack,  columns=['xgb1','lgb1','cat1','xgb3','lgb3','cat3'])
#     Flag_for6 =  1
#     print("used ALl model except 2nd stage ")







# from sklearn.preprocessing import StandardScaler
# stdc = StandardScaler()
# stdc.fit(train_stack1)
# train_stack = stdc.transform(train_stack1.values)
# test_stack = stdc.transform(test_stack1.values)




# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# def lr_schedule(epoch):
#     lr = 0.5e-2
#     if epoch > 140: lr *= 1e-3
#     elif epoch > 130: lr *= 0.5e-2
#     elif epoch > 110: lr *= 0.5e-1
#     elif epoch > 70: lr *= 1e-2
#     elif epoch > 40:  lr *= 1e-1
#     print('Learning rate: ', lr)
#     return lr
# lr_scheduler = LearningRateScheduler(lr_schedule)
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=1)
# checkpoint = ModelCheckpoint('chk_pt.h5', monitor='val_loss', period=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=2, min_lr=1e-6) 
# callbacks = [lr_scheduler, reduce_lr, checkpoint, early_stop]





# def rmse(y, y_pred):
#     return K.sqrt(K.mean(K.square(y-y_pred), axis=-1))

    
# nfolds=3
# # folds = StratifiedKFold(n_splits=nfolds,shuffle=True, random_state=15)
# folds = GroupKFold(n_splits=nfolds)#,shuffle=True, random_state=15)
# avg_train_kappa = 0
# avg_valid_kappa = 0
# batch_size=5000
# coeffs=None

# # x_test = get_keras_data(test_df, desc_embs[len(train_df):])
# #submission_df = test_df[["PetID"]]
# adoptions_keras = np.zeros((len(test_stack),))
# oof_train_keras = np.zeros((train_stack.shape[0]))
# #oof_xgb_3 = np.zeros(len(train))
# #predictions_xgb_3 = np.zeros(len(test))
# # oof_train_keras = np.zeros(train_stack.shape[0])
# oof_test_keras = np.zeros((test_stack.shape[0], nfolds))

# i =0 
# for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, X_train['AdoptionSpeed'].values, rescue_ids)):
#     print("fold n°{}".format(fold_))
#     trn_data, trn_y = train_stack[trn_idx], X_train['AdoptionSpeed'].iloc[trn_idx].values
#     val_data, val_y = train_stack[val_idx], X_train['AdoptionSpeed'].iloc[val_idx].values
#     model = get_model1(train_stack.shape[1])
#     print(trn_data.shape, trn_y.shape, val_data.shape, val_y.shape, train_stack.shape)
#     trn_y_ = 0.15*np.random.randn(trn_y.shape[0]) + trn_y
#     clr_tri = CyclicLR(base_lr=0.001, max_lr=0.008, step_size=len(train)//batch_size, mode="exp_range")#triangular2")
#     qwk_eval = QWKEvaluation(validation_data=(val_data, val_y.reshape(-1,1)), interval=1)
#     history = model.fit(trn_data, trn_y_.reshape(-1,1), batch_size=batch_size, 
#                         validation_data=(val_data, val_y.reshape(-1,1)), 
#                         epochs=150, 
# #                         callbacks=[clr_tri, qwk_eval, reduce_lr])
#                         callbacks = [qwk_eval, lr_scheduler, reduce_lr, checkpoint])#, early_stop])
#     model.load_weights('chk_pt.h5')
#     y_train_pred = model.predict(trn_data, batch_size=5000)
#     y_valid_pred = model.predict(val_data, batch_size=5000)
# #     avg_train_kappa += quadratic_weighted_kappa(list(eval_predict(trn_y.reshape(-1,1),y_train_pred)), list(trn_y))
#     #print('Train Kappa',quadratic_weighted_kappa(y_train_pred, y_train))
#     avg_valid_kappa += quadratic_weighted_kappa(list(eval_predict(val_y.reshape(-1,1),y_valid_pred)), list(val_y))
# #     #print('Valid Kappa',quadratic_weighted_kappa(y_valid_pred, y_valid))
# #     coeffs = eval_predict(val_y.reshape(-1,1), model.predict(val_data, batch_size=2500), ret_coeffs=True)
#     oof_test_keras[:, fold_] = model.predict(test_stack, batch_size=batch_size).reshape(-1)
#     oof_train_keras[val_idx] = y_valid_pred.reshape(-1)
        
# # print("\navg train kappa:", avg_train_kappa/nfolds)
# print("\navg valid kappa:", avg_valid_kappa/nfolds)







# print("\navg valid kappa:", avg_valid_kappa/nfolds)
# oof_train_keras.shape, oof_test_keras.shape



# import matplotlib.pyplot as plt

# f = plt.figure(figsize=(18,3))
# ax = f.add_subplot(121)
# ax2 = f.add_subplot(122)

# ax.plot(history.history['loss'][10:])
# ax.plot(history.history['val_loss'][10:])
# ax.set_title('Model loss')
# ax.set_xlabel('epoch')
# ax.legend(['train', 'valid'], loc='upper left')

# ax2.plot(history.history['lr'])
# ax2.set_title('Learning rate')
# ax2.set_xlabel('iteration')

# oof_train_keras1 = oof_train_keras
# oof_test_keras1 = oof_test_keras.mean(axis=1)



# def plot_pred(pred):
#     sns.distplot(pred, kde=True, hist_kws={'range': [0, 5]})

# plot_pred(oof_train_keras)
# plot_pred(oof_test_keras.mean(axis=1))

# optR = OptimizedRounder()
# optR.fit(oof_train_keras, X_train['AdoptionSpeed'].values)
# coefficients = optR.coefficients()
# valid_pred = optR.predict(oof_train_keras, coefficients)
# qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
# print("keras QWK = ", qwk)

# # optR = OptimizedRounder_auto()
# # optR.fit(oof_train_keras, X_train['AdoptionSpeed'].values)
# # coefficients = optR.coefficients()
# # valid_pred = optR.predict(oof_train_keras, coefficients)
# # qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
# # print("keras Auto QWK = ", qwk)


# coefficients_ = coefficients.copy()
# coefficients_[0] = 1.66
# coefficients_[1] = 2.13
# coefficients_[3] = 2.85
# train_predictions_keras = optR.predict(oof_train_keras, coefficients_).astype(np.int8)
# print(f'train pred distribution: {Counter(train_predictions_keras)}')
# test_predictions_keras = optR.predict(oof_test_keras.mean(axis=1), coefficients_).astype(np.int8)
# print(f'test pred distribution: {Counter(test_predictions_keras)}')

# submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions_keras})
# submission.to_csv('submission_keras.csv', index=False)
# submission.head()

# if qwk > result1 :
#     print("keras 1 is working better than ridge")
#     submission.to_csv('submission.csv', index=None)
#     result1 = qwk













def rmse(y, y_pred):
    return K.sqrt(K.mean(K.square(y-y_pred), axis=-1))

    
nfolds=10
folds = StratifiedKFold(n_splits=nfolds,shuffle=True, random_state=15)
avg_train_kappa = 0
avg_valid_kappa = 0
batch_size=5000
coeffs=None

# x_test = get_keras_data(test_df, desc_embs[len(train_df):])
#submission_df = test_df[["PetID"]]
# adoptions_keras = np.zeros((len(test_stack),))
oof_train_keras2 = np.zeros((train_stack.shape[0]))
#oof_xgb_3 = np.zeros(len(train))
#predictions_xgb_3 = np.zeros(len(test))
# oof_train_keras = np.zeros(train_stack.shape[0])
oof_test_keras2 = np.zeros((test_stack.shape[0], nfolds))

i =0 
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, X_train['AdoptionSpeed'].values)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], X_train['AdoptionSpeed'].iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], X_train['AdoptionSpeed'].iloc[val_idx].values
    model = get_model()
    clr_tri = CyclicLR(base_lr=2e-3, max_lr=4e-2, step_size=len(train)//batch_size, mode="triangular2")
    qwk_eval = QWKEvaluation(validation_data=(val_data, val_y.reshape(-1,1)), interval=1)
    history = model.fit(trn_data, trn_y.reshape(-1,1), batch_size=batch_size, 
                        validation_data=(val_data, val_y.reshape(-1,1)), 
                        epochs=50, callbacks=[clr_tri, qwk_eval])
    model.load_weights('checkpoint.h5')
    y_train_pred = model.predict(trn_data, batch_size=5000)
    y_valid_pred = model.predict(val_data, batch_size=5000)
#     avg_train_kappa += quadratic_weighted_kappa(list(eval_predict(trn_y.reshape(-1,1),y_train_pred)), list(trn_y))
    #print('Train Kappa',quadratic_weighted_kappa(y_train_pred, y_train))
    avg_valid_kappa += quadratic_weighted_kappa(list(eval_predict(val_y.reshape(-1,1),y_valid_pred)), list(val_y))
#     #print('Valid Kappa',quadratic_weighted_kappa(y_valid_pred, y_valid))
#     coeffs = eval_predict(val_y.reshape(-1,1), model.predict(val_data, batch_size=2500), ret_coeffs=True)
    oof_test_keras2[:, fold_] = model.predict(test_stack, batch_size=batch_size).reshape(-1)
    oof_train_keras[val_idx] = y_valid_pred.reshape(-1)
        
# print("\navg train kappa:", avg_train_kappa/nfolds)
print("\navg valid kappa:", avg_valid_kappa/nfolds)
