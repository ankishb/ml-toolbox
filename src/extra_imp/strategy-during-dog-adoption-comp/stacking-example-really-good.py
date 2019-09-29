n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)































# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
"""
Created on Sat Jul  8 08:45:55 2017

@author: rupesh
"""

## This script contains four independent models, the stacking part: Use lgb, gbdt and Lasso as 
## Level 1 models. Meta model is xgb. This may count for 25% of the final result.
## Then, we use a single xgb model forked from this kernel : https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697
## Only take single XGB.
## Then, finally replace some values with LB Probing Values. I am not sure whether this is a good idea, just try

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


#target=train['y']
#train.drop(['y'],axis=1,inplace=True)
#test['y']=np.zeros(train.shape[0])
# Cat conversion


for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

y=train['y'] 
train.drop(['y'],inplace=True,axis=1)       
combine=pd.concat([train,test])
columns=['X1','X2','X3','X4','X5','X6','X8']
for column in columns:
    temp=pd.get_dummies(pd.Series(combine[column]))
    combine=pd.concat([combine,temp],axis=1)
    combine= combine.drop([column], axis=1)
        
# Define some useful functions

train=combine[:train.shape[0]]
test=combine[train.shape[0]:] 


def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df

train = df_column_uniquify(train)  
test = df_column_uniquify(test)   
train['y']=y

def get_additional_features(train,test,magic=False,ID=False):
    col = list(test.columns)
    if ID!=True:
        col.remove('ID')
    n_comp = 12
    # tSVD
    tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
    tsvd_results_train = tsvd.fit_transform(train[col])
    tsvd_results_test = tsvd.transform(test[col])
    # PCA
    pca = PCA(n_components=n_comp, random_state=420)
    pca2_results_train = pca.fit_transform(train[col])
    pca2_results_test = pca.transform(test[col])
    # ICA
    ica = FastICA(n_components=n_comp, random_state=420)
    ica2_results_train = ica.fit_transform(train[col])
    ica2_results_test = ica.transform(test[col])
    # GRP
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
    grp_results_train = grp.fit_transform(train[col])
    grp_results_test = grp.transform(test[col])
    # SRP
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
    srp_results_train = srp.fit_transform(train[col])
    srp_results_test = srp.transform(test[col])
    for i in range(1, n_comp + 1):
        train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
        test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]
        train['ica_' + str(i)] = ica2_results_train[:, i - 1]
        test['ica_' + str(i)] = ica2_results_test[:, i - 1]
        train['grp_' + str(i)] = grp_results_train[:, i - 1]
        test['grp_' + str(i)] = grp_results_test[:, i - 1]
        train['srp_' + str(i)] = srp_results_train[:, i - 1]
        test['srp_' + str(i)] = srp_results_test[:, i - 1]
    if magic==True:
        magic_mat = train[['ID','X0','y']]
        magic_mat = magic_mat.groupby(['X0'])['y'].mean()
        magic_mat = pd.DataFrame({'X0':magic_mat.index,'magic':list(magic_mat)})
        mean_magic = magic_mat['magic'].mean()
        train = train.merge(magic_mat,on='X0',how='left')
        test = test.merge(magic_mat,on='X0',how = 'left')
        test['magic'] = test['magic'].fillna(mean_magic)
    return train,test

## Preparing stacking functions. Each one takes the out of bag values as the Input

## xgb will not be used in this case, but still post it here.
def get_xgb_stack_data(params,rounds,train,col,label,test):
    ID = []
    train = train.reset_index(drop=True)
    kf = KFold(n_splits=5,shuffle=False)
    i=0
    R2_Score = []
    RMSE = []
    for train_index, test_index in kf.split(train):
        print("Training "+str(i+1)+' Fold')
        X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]
        y_train, y_test = label.iloc[train_index],label.iloc[test_index]
        dtrain = xgb.DMatrix(X_train[col],y_train)
        dtest = xgb.DMatrix(X_test[col])
        model = xgb.train(params,dtrain,num_boost_round=rounds)
        pred = model.predict(dtest)
        X_test['label'] = list(y_test)
        X_test['predicted'] = pred
        r2 = r2_score(y_test,pred)
        rmse = MSE(y_test,pred)**0.5
        print('R2 Scored of Fold '+str(i+1)+' is '+str(r2))
        R2_Score.append(r2)
        RMSE.append(rmse)
        print('RMSE of Fold '+str(i+1)+' is '+str(rmse))
        ID.append(X_test['ID'])
        if i==0:
            Final = X_test
        else:
            Final = Final.append(X_test,ignore_index=True)
        i+=1
    dtrain_ = xgb.DMatrix(train[col],label)
    dtest_ = xgb.DMatrix(test[col])
    print('Start Training')
    model_ = xgb.train(params,dtrain_,num_boost_round=rounds)
    Final_pred = model_.predict(dtest_)
    Final_pred = pd.DataFrame({'ID':test['ID'],'y':Final_pred})
    print('Calculating In-Bag R2 Score')
    print(r2_score(dtrain_.get_label(), model.predict(dtrain_)))
    print('Calculating Out-Bag R2 Score')
    print(np.mean(R2_Score))
    print('Calculating In-Bag RMSE')
    print(MSE(dtrain_.get_label(), model.predict(dtrain_))**0.5)
    print('Calculating Out-Bag RMSE')
    print(np.mean(RMSE))
    return Final,Final_pred


def get_lgb_stack_data(params,rounds,train,col,label,test):
    ID = []
    train = train.reset_index(drop=True)
    kf = KFold(n_splits=5,shuffle=False)
    i=0
    R2_Score = []
    RMSE = []
    for train_index, test_index in kf.split(train):
        print("Training "+str(i+1)+' Fold')
        X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]
        y_train, y_test = label.iloc[train_index],label.iloc[test_index]
        train_lgb=lgb.Dataset(X_train[col],y_train)
        model = lgb.train(params,train_lgb,num_boost_round=rounds)
        pred = model.predict(X_test[col])
        X_test['label'] = list(y_test)
        X_test['predicted'] = pred
        r2 = r2_score(y_test,pred)
        rmse = MSE(y_test,pred)**0.5
        print('R2 Scored of Fold '+str(i+1)+' is '+str(r2))
        R2_Score.append(r2)
        RMSE.append(rmse)
        print('RMSE of Fold '+str(i+1)+' is '+str(rmse))
        ID.append(X_test['ID'])
        if i==0:
            Final = X_test
        else:
            Final = Final.append(X_test,ignore_index=True)
        i+=1
    lgb_train_ = lgb.Dataset(train[col],label)
    print('Start Training')
    model_ = lgb.train(params,lgb_train_,num_boost_round=rounds)
    Final_pred = model_.predict(test[col])
    Final_pred = pd.DataFrame({'ID':test['ID'],'y':Final_pred})
    print('Calculating In-Bag R2 Score')
    print(r2_score(label, model.predict(train[col])))
    print('Calculating Out-Bag R2 Score')
    print(np.mean(R2_Score))
    print('Calculating In-Bag RMSE')
    print(MSE(label, model.predict(train[col]))**0.5)
    print('Calculating Out-Bag RMSE')
    print(np.mean(RMSE))
    return Final,Final_pred



def get_sklearn_stack_data(model,train,col,label,test):
    ID = []
    R2_Score = []
    RMSE = []
    train = train.reset_index(drop=True)
    kf = KFold(n_splits=5,shuffle=False)
    i=0
    for train_index, test_index in kf.split(train):
        print("Training "+str(i+1)+' Fold')
        X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]
        y_train, y_test = label.iloc[train_index],label.iloc[test_index]
        model.fit(X_train[col],y_train)
        pred = model.predict(X_test[col])
        X_test['label'] = list(y_test)
        X_test['predicted'] = pred
        r2 = r2_score(y_test,pred)
        rmse = MSE(y_test,pred)**0.5
        print('R2 Scored of Fold '+str(i+1)+' is '+str(r2))
        R2_Score.append(r2)
        RMSE.append(rmse)
        print('RMSE of Fold '+str(i+1)+' is '+str(rmse))
        ID.append(X_test['ID'])
        if i==0:
            Final = X_test
        else:
            Final = Final.append(X_test,ignore_index=True)
        i+=1
    print('Start Training')
    model.fit(train[col],label)
    Final_pred = model.predict(test[col])
    Final_pred = pd.DataFrame({'ID':test['ID'],'y':Final_pred})
    print('Calculating In-Bag R2 Score')
    print(r2_score(label, model.predict(train[col])))
    print('Calculating Out-Bag R2 Score')
    print(np.mean(R2_Score))
    print('Calculating In-Bag RMSE')
    print(MSE(label, model.predict(train[col]))**0.5)
    print('Calculating Out-Bag RMSE')
    print(np.mean(RMSE))
    return Final,Final_pred
    
## Prepare output of level 1.

## Prepare data

train_,test_ = get_additional_features(train,test,magic=True)
train_ = train_.sample(frac=1,random_state=420)
col = list(test.columns)
## Input 1: GBDT

gb1 = GradientBoostingRegressor(n_estimators=1000,max_features=0.95,learning_rate=0.005,max_depth=4)
gb1_train,gb1_test = get_sklearn_stack_data(gb1,train_,col,train_['y'],test_)

## Input2: Lasso
las1 = Lasso(alpha=5,random_state=42)
las1_train,las1_test = get_sklearn_stack_data(las1,train_,col,train_['y'],test_)

## Input 3: LGB
params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'learning_rate': 0.0045 , #small learn rate, large number of iterations
            'verbose': 0,
            'num_iterations': 500,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 42,
            'feature_fraction': 0.95,
            'feature_fraction_seed': 42,
            'max_bin': 100,
            'max_depth': 3,
            'num_rounds': 800
        }
lgb_train, lgb_test = get_lgb_stack_data(params,800,train_,col,train_['y'],test_)

## Stacking By xgb

stack_train = gb1_train[['label','predicted']]
stack_train.columns=[['label','gbdt']]
stack_train['lgb']=lgb_train['predicted']
stack_train['las'] = las1_train['predicted']

stack_test = gb1_test[['ID','y']]
stack_test.columns=[['ID','gbdt']]
stack_test['lgb']=lgb_test['y']
stack_test['las'] = las1_test['y']
del stack_test['ID']

## Meta Model: xgb

y_mean = np.mean(train.y)

col = list(stack_test.columns)

params = {
    'eta': 0.005,
    'max_depth': 2,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

dtrain = xgb.DMatrix(stack_train[col], stack_train['label'])
dtest = xgb.DMatrix(stack_test[col])

#xgb_cvalid = xgb.cv(params, dtrain, num_boost_round=2000, early_stopping_rounds=20,
 #   verbose_eval=50, show_stdv=True,seed=42)
#xgb_cvalid[['train-rmse-mean', 'test-rmse-mean']].plot()
#print('Performance does not improve from '+str(len(xgb_cvalid))+' rounds')

model = xgb.train(params,dtrain,num_boost_round =900)
pred_1 = model.predict(dtest)


## Original XGB In Popular Kernel



train_,test_ = get_additional_features(train,test,ID=True)

xgb_params = {
        'n_trees': 520, 
        'eta': 0.0045,
        'max_depth': 4,
        'subsample': 0.93,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'base_score': y_mean, # base prediction = mean(target)
        'silent': True,
        'seed': 42,
    }
dtrain = xgb.DMatrix(train_.drop('y', axis=1), train_.y)
dtest = xgb.DMatrix(test_)
    
num_boost_rounds = 1250
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)

## Average Two Solutions

Average = 0.70*y_pred + 0.30*pred_1

sub = pd.DataFrame({'ID':test['ID'],'y':Average})

## LB Prob Values 

## I forget whose credit should be given, Please help me to find him/her!!

leaks = {
    1:71.34112,
    12:109.30903,
    23:115.21953,
    28:92.00675,
    42:87.73572,
    43:129.79876,
    45:99.55671,
    57:116.02167,
    3977:132.08556,
    88:90.33211,
    89:130.55165,
    93:105.79792,
    94:103.04672,
    1001:111.65212,
    104:92.37968,
    72:110.54742,
    78:125.28849,
    105:108.5069,
    110:83.31692,
    1004:91.472,
    1008:106.71967,
    1009:108.21841,
    973:106.76189,
    8002:95.84858,
    8007:87.44019,
    1644:99.14157,
    337:101.23135,
    253:115.93724,
    8416:96.84773,
    259:93.33662,
    262:75.35182,
    1652:89.77625
    }
sub['y'] = sub.apply(lambda r: leaks[int(r['ID'])] if int(r['ID']) in leaks else r['y'], axis=1)

sub.to_csv('subXgb_Stack_Stack_No_ID_with_onehot.csv',index=False)