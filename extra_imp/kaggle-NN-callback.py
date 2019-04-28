

# ################## Good Example ##################					

from keras.layers import Dense, Flatten, Input, Dropout
from keras import Model, regularizers
from keras.optimizers import Adam
def get_model1():
    inp = Input(shape=(6,))
    x = Dense(150, activation='relu')(inp)
    x = Dropout(0.25)(x)
    x = Dense(150, activation='relu')(x)#, activity_regularizer=regularizers.l1(10e-5))(x)
    x = Dropout(0.25)(x)
    x = Dense(150, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation='linear')(x)
    model = Model(inp, x)
    opt = Adam()
    model.compile(optimizer=opt, loss='mse',)
    return model



from keras.callbacks import EarlyStopping, ReduceLROnPlateau
early = EarlyStopping( monitor='val_loss',verbose=1, patience=3)
reduce_lr = ReduceLROnPlateau( monitor='val_loss', factor=0.1,
                               patience=1, verbose=1, mode='auto', 
                               min_delta=0.0001, min_lr=0,)

from keras.callbacks import *

class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
    
class QWKEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.history = []
        self.X_val, self.y_val = validation_data
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=1000, verbose=0)
            y_pred = eval_predict(self.y_val, y_pred)
            
            score = quadratic_weighted_kappa(list(self.y_val.reshape(-1,)), list(y_pred))
            print("QWK - epoch: %d - score: %.6f \n" % (epoch+1, score))
            self.history.append(score)
            if score >= max(self.history): self.model.save('checkpoint.h5')

def eval_predict(y=[], y_pred=[], coeffs=None, ret_coeffs=False):
    optR = OptimizedRounder()
    if not coeffs:
        optR.fit(y_pred.reshape(-1,), y.reshape(-1,))
        coeffs = optR.coefficients()
    if ret_coeffs: return optR.coefficients()
    return optR.predict(y_pred.reshape(-1,), coeffs)


def rmse(y, y_pred):
    return K.sqrt(K.mean(K.square(y-y_pred), axis=-1))

    
nfolds=10
folds = StratifiedKFold(n_splits=nfolds,shuffle=True, random_state=15)
avg_train_kappa = 0
avg_valid_kappa = 0
batch_size=5000
coeffs=None

adoptions_keras = np.zeros((len(test_stack),))
oof_train_keras = np.zeros((train_stack.shape[0]))
oof_test_keras = np.zeros((test_stack.shape[0], nfolds))

i =0 
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, X_train['AdoptionSpeed'].values)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], X_train['AdoptionSpeed'].iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], X_train['AdoptionSpeed'].iloc[val_idx].values
    model = get_model1()
    clr_tri = CyclicLR(base_lr=2e-3, max_lr=4e-2, step_size=len(train)//batch_size, mode="triangular2")
    qwk_eval = QWKEvaluation(validation_data=(val_data, val_y.reshape(-1,1)), interval=1)
    history = model.fit(trn_data, trn_y.reshape(-1,1), batch_size=batch_size, 
                        validation_data=(val_data, val_y.reshape(-1,1)), 
                        epochs=60, callbacks=[clr_tri, qwk_eval])
    model.load_weights('checkpoint.h5')
    y_train_pred = model.predict(trn_data, batch_size=5000)
    y_valid_pred = model.predict(val_data, batch_size=5000)

    avg_valid_kappa += quadratic_weighted_kappa(list(eval_predict(val_y.reshape(-1,1),y_valid_pred)), list(val_y))
    oof_test_keras[:, fold_] = model.predict(test_stack, batch_size=batch_size).reshape(-1)
    oof_train_keras[val_idx] = y_valid_pred.reshape(-1)
        
print("\navg valid kappa:", avg_valid_kappa/nfolds)





# ################## Another Good Example ##################					

#go here, it's easier to understand callbacks reading keras source code:
#   https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L838
#   https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L1040

from sklearn.metrics import roc_auc_score
class GiniWithEarlyStopping(keras.callbacks.Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=1024):
        #print("self vars: ",vars(self))  #uncomment and discover some things =)
        
        # FROM EARLY STOP
        super(GiniWithEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size=predict_batch_size
    
    def on_batch_begin(self, batch, logs={}):
        # if(self.verbose > 1):
            # if(batch!=0):
                # print("")
            # print("Hi! on_batch_begin() , batch=",batch,",logs:",logs)
		pass 
		
    def on_batch_end(self, batch, logs={}):
        # if(self.verbose > 1):
            # print("Hi! on_batch_end() , batch=",batch,",logs:",logs)
		pass 
		
    def on_train_begin(self, logs={}):
        # if(self.verbose > 1):
            # print("Hi! on_train_begin() ,logs:",logs)

        # FROM EARLY STOP
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
    
    def on_train_end(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_end() ,logs:",logs)

        # FROM EARLY STOP
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch ',self.stopped_epoch,': GiniEarlyStopping')
    
    def on_epoch_begin(self, epoch, logs={}):
        if(self.verbose > 1):
            print("Hi! on_epoch_begin() , epoch=",epoch,",logs:",logs)

    def on_epoch_end(self, epoch, logs={}):
        if(self.validation_data):
            y_hat_val=self.model.predict(self.validation_data[0],batch_size=self.predict_batch_size)
            
        if(self.verbose > 1):
            print("Hi! on_epoch_end() , epoch=",epoch,",logs:",logs)
        
            print("    GINI Callback:")
            if(self.validation_data):
                print('        validation_data.inputs       : ',np.shape(self.validation_data[0]))
                print('        validation_data.targets      : ',np.shape(self.validation_data[1]))
                print("        roc_auc_score(y_real,y_hat)  : ",roc_auc_score(self.validation_data[1], y_hat_val ))
                print("        gini_normalized(y_real,y_hat): ",gini_normalized(self.validation_data[1], y_hat_val))
                print("        roc_auc_scores*2-1           : ",roc_auc_score(self.validation_data[1], y_hat_val)*2-1)
        
            print('    Logs (others metrics):',logs)
        # FROM EARLY STOP
        if(self.validation_data):
            if (self.verbose == 1):
                print("\n GINI Callback:",gini_normalized(self.validation_data[1], y_hat_val))
            current = gini_normalized(self.validation_data[1], y_hat_val)
            
            # to logs (scores) and use others callbacks too....
            # logs['gini_val']=current
            
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True


Option 2 - magic, Include metric in logs dictionary

example with Roc

from sklearn.metrics import roc_auc_score
class RocAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size=predict_batch_size
        self.include_on_batch=include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if(self.include_on_batch):
            logs['roc_auc_val']=float('-inf')
            if(self.validation_data):
                logs['roc_auc_val']=roc_auc_score(self.validation_data[1], 
                                                  self.model.predict(self.validation_data[0],
                                                                     batch_size=self.predict_batch_size))

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val']=float('-inf')
        if(self.validation_data):
            logs['roc_auc_val']=roc_auc_score(self.validation_data[1], 
                                              self.model.predict(self.validation_data[0],
                                                                 batch_size=self.predict_batch_size))

from keras.callbacks import EarlyStopping
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# reduce train size, just to this kernel example
t=train[0:1000]
# batch_size=500 ~= 2 batchs
estimator = KerasClassifier(build_fn=model_relu1, nb_epoch=3, batch_size=500, verbose=1)

cb = [
    RocAucMetricCallback(), # include it before EarlyStopping!
    EarlyStopping(monitor='roc_auc_val',patience=1, verbose=2) 
]

estimator.fit(t[col].values,t['target'],epochs=100,validation_split=.2,callbacks=cb)









					
					
					
					
# ################## Another Good Example ##################					
					
import numpy as np
np.random.seed()
import pandas as pd
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.wrappers.scikit_learn import KerasClassifier

# Using TensorFlow backend.

# This callback is very important. It calculates roc_auc and gini values so they can be monitored during the run. Also, it creates a log of those parameters so that they can be used for early stopping. A tip of the hat to Roberto and this kernel for helping me figure out the latter.

# Note that this callback in combination with early stopping doesn't print well if you are using verbose=1 (moving arrow) during fitting. I recommend that you use verbose=2 in fitting.

class roc_auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_proba(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = ( roc_auc_score(self.y, y_pred) * 2 ) - 1

        y_pred_val = self.model.predict_proba(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = ( roc_auc_score(self.y_val, y_pred_val) * 2 ) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))), end=10*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

		
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))


folds = 4
runs = 2

cv_LL = 0
cv_AUC = 0
cv_gini = 0
fpred = []
avpred = []
avreal = []
avids = []

Loading data. Converting "categorical" variables, even though in this dataset they are actually numeric.

# Load data set and target values
train, target, test, tr_ids, te_ids = load_data()
n_train = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)
col_to_drop = train.columns[train.columns.str.endswith('_cat')]
col_to_dummify = train.columns[train.columns.str.endswith('_cat')].astype(str).tolist()

for col in col_to_dummify:
    dummy = pd.get_dummies(train_test[col].astype('category'))
    columns = dummy.columns.astype(str).tolist()
    columns = [col + '_' + w for w in columns]
    dummy.columns = columns
    train_test = pd.concat((train_test, dummy), axis=1)

train_test.drop(col_to_dummify, axis=1, inplace=True)
train_test_scaled, scaler = scale_data(train_test)
train = train_test_scaled[:n_train, :]
test = train_test_scaled[n_train:, :]
print('\n Shape of processed train data:', train.shape)
print(' Shape of processed test data:', test.shape)

patience = 10
batchsize = 128

skf = StratifiedKFold(n_splits=folds, random_state=1001)
starttime = timer(None)
for i, (train_index, test_index) in enumerate(skf.split(train, target)):
    start_time = timer(None)
    X_train, X_val = train[train_index], train[test_index]
    y_train, y_val = target[train_index], target[test_index]
    train_ids, val_ids = tr_ids[train_index], tr_ids[test_index]
    
# This is where we define and compile the model. These parameters are not optimal, as they were chosen 
# to get a notebook to complete in 60 minutes. Other than leaving BatchNormalization and last sigmoid 
# activation alone, virtually everything else can be optimized: number of neurons, types of initializers, 
# activation functions, dropout values. The same goes for the optimizer at the end.

#########
# Never move this model definition to the beginning of the file or anywhere else outside of this loop. 
# The model needs to be initialized anew every time you run a different fold. If not, it will continue 
# the training from a previous model, and that is not what you want.
#########

    # This definition must be within the for loop or else it will continue training previous model
    def baseline_model():
        model = Sequential()
        model.add( Dense(200, input_dim=X_train.shape[1], kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(100, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(50, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        model.add(Dense(25, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(optimizer='adam', metrics = ['accuracy'], loss='binary_crossentropy')

        return model

# This is where we repeat the runs for each fold. If you choose runs=1 above, it will run a 
# regular N-fold procedure.

#########
# It is important to leave the call to random seed here, so each run starts with a different seed.
#########

    for run in range(runs):
        print('\n Fold %d - Run %d\n' % ((i + 1), (run + 1)))
        np.random.seed()

# Lots to unpack here.

# The first callback prints out roc_auc and gini values at the end of each epoch. It must be listed 
# before the EarlyStopping callback, which monitors gini values saved in the previous callback. Make 
# sure to set the mode to "max" because the default value ("auto") will not handle gini properly 
# (it will act as if the model is not improving even when roc/gini go up).

# CSVLogger creates a record of all iterations. Not really needed but it doesn't hurt to have it.

# ModelCheckpoint saves a model each time gini improves. Its mode also must be set to "max" for reasons 
# explained above.

        callbacks = [
            roc_auc_callback(training_data=(X_train, y_train),validation_data=(X_val, y_val)),  # call this before EarlyStopping
            EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=1),
            CSVLogger('keras-5fold-run-01-v1-epochs.log', separator=',', append=False),
            ModelCheckpoint(
                    'keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check',
                    monitor='norm_gini_val', mode='max', # mode must be set to max or Keras will be confused
                    save_best_only=True,
                    verbose=1)
        ]

        nnet = KerasClassifier(
            build_fn=baseline_model,
            epochs=3,
            batch_size=batchsize,
            validation_data=(X_val, y_val),
            verbose=2,
            shuffle=True,
            callbacks=callbacks)

        fit = nnet.fit(X_train, y_train)
        
# We want the best saved model - not the last one where the training stopped. So we delete the old 
# model instance and load the model from the last saved checkpoint. Next we predict values both for 
# validation and test data, and create a summary of parameters for each run.

        del nnet
        nnet = load_model('keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')
        scores_val_run = nnet.predict_proba(X_val, verbose=0)
        LL_run = log_loss(y_val, scores_val_run)
        print('\n Fold %d Run %d Log-loss: %.5f' % ((i + 1), (run + 1), LL_run))
        AUC_run = roc_auc_score(y_val, scores_val_run)
        print(' Fold %d Run %d AUC: %.5f' % ((i + 1), (run + 1), AUC_run))
        print(' Fold %d Run %d normalized gini: %.5f' % ((i + 1), (run + 1), AUC_run*2-1))
        y_pred_run = nnet.predict_proba(test, verbose=0)
        if run > 0:
            scores_val = scores_val + scores_val_run
            y_pred = y_pred + y_pred_run
        else:
            scores_val = scores_val_run
            y_pred = y_pred_run
            
# We average all runs from the same fold and provide a parameter summary for each fold. Unless something 
# is wrong, the numbers printed here should be better than any of the individual runs.

    scores_val = scores_val / runs
    y_pred = y_pred / runs
    LL = log_loss(y_val, scores_val)
    print('\n Fold %d Log-loss: %.5f' % ((i + 1), LL))
    AUC = roc_auc_score(y_val, scores_val)
    print(' Fold %d AUC: %.5f' % ((i + 1), AUC))
    print(' Fold %d normalized gini: %.5f' % ((i + 1), AUC*2-1))
    timer(start_time)
    
# We add up predictions on the test data for each fold. Create out-of-fold predictions for validation data.

    if i > 0:
        fpred = pred + y_pred
        avreal = np.concatenate((avreal, y_val), axis=0)
        avpred = np.concatenate((avpred, scores_val), axis=0)
        avids = np.concatenate((avids, val_ids), axis=0)
    else:
        fpred = y_pred
        avreal = y_val
        avpred = scores_val
        avids = val_ids
    pred = fpred
    cv_LL = cv_LL + LL
    cv_AUC = cv_AUC + AUC
    cv_gini = cv_gini + (AUC*2-1)


LL_oof = log_loss(avreal, avpred)
print('\n Average Log-loss: %.5f' % (cv_LL/folds))
print(' Out-of-fold Log-loss: %.5f' % LL_oof)
AUC_oof = roc_auc_score(avreal, avpred)
print('\n Average AUC: %.5f' % (cv_AUC/folds))
print(' Out-of-fold AUC: %.5f' % AUC_oof)
print('\n Average normalized gini: %.5f' % (cv_gini/folds))
print(' Out-of-fold normalized gini: %.5f' % (AUC_oof*2-1))
score = str(round((AUC_oof*2-1), 5))
timer(starttime)
mpred = pred / folds
















################################## KAGGLE LIBERTY CHALLENGE ######################################
https://github.com/Far0n/kaggle-lmgpip/blob/master/learn.py

https://github.com/ffyu/Kaggle-Liberty-Mutual-Competition





#go here, it's easier to understand callbacks reading keras source code:
#   https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L838
#   https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L1040

from sklearn.metrics import roc_auc_score
class GiniWithEarlyStopping(keras.callbacks.Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=1024):
        #print("self vars: ",vars(self))  #uncomment and discover some things =)
        
        # FROM EARLY STOP
        super(GiniWithEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size=predict_batch_size
    
    def on_batch_begin(self, batch, logs={}):
        if(self.verbose > 1):
            if(batch!=0):
                print("")
            print("Hi! on_batch_begin() , batch=",batch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)
    
    def on_batch_end(self, batch, logs={}):
        if(self.verbose > 1):
            print("Hi! on_batch_end() , batch=",batch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)
    
    def on_train_begin(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_begin() ,logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
    
    def on_train_end(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_end() ,logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch ',self.stopped_epoch,': GiniEarlyStopping')
    
    def on_epoch_begin(self, epoch, logs={}):
        if(self.verbose > 1):
            print("Hi! on_epoch_begin() , epoch=",epoch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

    def on_epoch_end(self, epoch, logs={}):
        if(self.validation_data):
            y_hat_val=self.model.predict(self.validation_data[0],batch_size=self.predict_batch_size)
            
        if(self.verbose > 1):
            print("Hi! on_epoch_end() , epoch=",epoch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)
        
        #i didn't found train data to check gini on train set (@TODO HERE)
        # from source code of Keras: https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L1127
        # for cbk in callbacks:
        #     cbk.validation_data = val_ins
        # Probably we will need to change keras... 
        # 
        
            print("    GINI Callback:")
            if(self.validation_data):
                print('        validation_data.inputs       : ',np.shape(self.validation_data[0]))
                print('        validation_data.targets      : ',np.shape(self.validation_data[1]))
                print("        roc_auc_score(y_real,y_hat)  : ",roc_auc_score(self.validation_data[1], y_hat_val ))
                print("        gini_normalized(y_real,y_hat): ",gini_normalized(self.validation_data[1], y_hat_val))
                print("        roc_auc_scores*2-1           : ",roc_auc_score(self.validation_data[1], y_hat_val)*2-1)

            print('    Logs (others metrics):',logs)
        # FROM EARLY STOP
        if(self.validation_data):
            if (self.verbose == 1):
                print("\n GINI Callback:",gini_normalized(self.validation_data[1], y_hat_val))
            current = gini_normalized(self.validation_data[1], y_hat_val)
            
            # we can include an "gambiarra" (very usefull brazilian portuguese word)
            # to logs (scores) and use others callbacks too....
            # logs['gini_val']=current
            
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True


        
                    
                    
    def rough():
        pass
    

def rough():
    pass

    
    
    

    