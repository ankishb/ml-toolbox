

import numpy as np
import pandas as pd
import gc

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC

def regression_model(reg_name):

    if clf_name == 'bayesian_ridge':
        params = {'alpha1' : 0.1, 
                  'alpha2' : 0.1,
                  'lambda1': 0.1,
                  'lambda2': 0.1}   
    else:
        params = {'alpha': 0.1}   


    if reg_name == 'bayesian_ridge':
        reg = BayesianRidge(
            n_iter=300, tol=0.001, compute_score=False,
            alpha_1=params['alpha1'], alpha_2=params['alpha2'], 
            lambda_1=params['lambda1'], lambda_2=params['lambda2'],  
            fit_intercept=True, normalize=True, verbose=False)

    elif reg_name is 'linear':
        reg = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)

    elif reg_name is 'lasso':
        reg = Lasso(
            alpha=params['alpha'], fit_intercept=True, normalize=True, 
            positive=False, random_state=1234, selection='cyclic')

    
    elif reg_name is 'elastic_net':
        reg = ElasticNet(
            alpha=params['alpha'], l1_ratio=0.5, fit_intercept=True, 
            normalize=True, random_state=1234, selection='cyclic')
    
    elif reg_name is 'ridge':
        reg = Ridge(
            alpha=params['alpha'], fit_intercept=True, normalize=True, 
            max_iter=500, random_state=1234)

    elif reg_name is 'kernel_ridge':
        reg = KernelRidge(
            alpha=params['alpha'], kernel=kernel_func, gamma=None, 
            degree=degree, coef0=1, kernel_params=None)
    
    elif reg_name is 'svm':
        reg = SVR(
            kernel=kernel_func, degree=degree, coef0=0.0, tol=0.001, 
            C=params['alpha'], epsilon=0.1, shrinking=True, cache_size=200, 
            verbose=0, max_iter=500)
    
    else:
        raise Exception('only [bayesian_ridge, lasso, elastic_net, ridge, kernel_ridge, svm] are supported')
    
    return reg

def classifier_model(clf_name):
    params = {'alpha':0.1}

    if clf_name == 'passive_agg':

        clf = PassiveAggressiveClassifier(
            C=params['alpha'], fit_intercept=True, max_iter=None, tol=None, 
            early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, 
            shuffle=True, verbose=0, n_jobs=-1, random_state=1234, loss='hinge',
            class_weight='balanced', average=False, n_iter=100)

    elif clf_name is 'ridge':
        clf = RidgeClassifier(
            alpha=params['alpha'], fit_intercept=True, normalize=True, 
            class_weight='balanced', random_state=1234)

    elif clf_name is 'logistic':
        clf = LogisticRegression(
            penalty='l2', dual=False, C=params['alpha'], fit_intercept=True, 
            intercept_scaling=1, class_weight=None, random_state=1234, 
            max_iter=100, multi_class='warn', verbose=0, n_jobs=-1)

    elif clf_name is 'svm':
        clf = SVC(
            C=params['alpha'], kernel=kernel_func, degree=degree, coef0=0.0, 
            shrinking=True, probability=False, tol=0.001, cache_size=200, 
            class_weight='balanced', verbose=False, max_iter=200, 
            decision_function_shape='ovr', random_state=1234)
    
    else:
        raise Exception('only [passive_agg, ridge, logistic, svm] are supported')
    
   return clf



def run_hyperopt_clf(train_df, target, max_evals, clf_name, kernel_func='linear', degree=3):
    """ Return best hyperparameter (mainly regularization parameters)
    Args:
      train_df, target
      max_evals: Total number of iteration to perform for bayesian optimization
      clf_name: name of regression model. [passive_agg, ridge, logistic, svm]
      kernel_func: [linear, rbf, poly]
      degree: degree of polynomial kernel
    example:
      best, tpe_trials = run_hyperopt(train_df, target, 20, 'kernel_ridge', kernel_func='poly')
    """
    def bayesian_opt(params):
        random_seed = 1234
        n_splits = 3

        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        oof_lgb = np.zeros(len(train_df))

        score_cv = []

        for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):

            y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
            X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]

            if clf_name == 'passive_agg':
            
                clf = PassiveAggressiveClassifier(
                    C=params['alpha'], fit_intercept=True, max_iter=None, tol=None, 
                    early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, 
                    shuffle=True, verbose=0, n_jobs=-1, random_state=1234, loss='hinge',
                    class_weight='balanced', average=False, n_iter=100)
            
            elif clf_name is 'ridge':
                clf = RidgeClassifier(
                    alpha=params['alpha'], fit_intercept=True, normalize=True, 
                    class_weight='balanced', random_state=1234)
            
            elif clf_name is 'logistic':
                clf = LogisticRegression(
                    penalty='l2', dual=False, C=params['alpha'], fit_intercept=True, 
                    intercept_scaling=1, class_weight=None, random_state=1234, 
                    max_iter=100, multi_class='warn', verbose=0, n_jobs=-1)
            
            elif clf_name is 'svm':
                clf = SVC(
                    C=params['alpha'], kernel=kernel_func, degree=degree, coef0=0.0, 
                    shrinking=True, probability=False, tol=0.001, cache_size=200, 
                    class_weight='balanced', verbose=False, max_iter=200, 
                    decision_function_shape='ovr', random_state=1234)
            
            else:
              raise Exception('only [passive_agg, ridge, logistic, svm] are supported')
            
            clf.fit(X_train, y_train)
            score = clf.score(X_valid, y_valid)
    #         score = roc_auc_score(y_valid, oof)
            score_cv.append(score)

        return -np.mean(score_cv)

    bayesian_params = {'alpha': hp.uniform('alpha', 0.01, 1000),}   
    
    trials = Trials()
    results = fmin(bayesian_opt, bayesian_params, algo=tpe.suggest, 
                   trials=trials, max_evals=max_evals)
        
    return results, trials 




def run_hyperopt_reg_modified(X_train_, X_valid_, y_train_, y_valid_, max_evals, clf_name, std_norm=True, kernel_func='linear', degree=3):
    """ Return best hyperparameter (mainly regularization parameters)
    Args:
      train_df, target
      max_evals: Total number of iteration to perform for bayesian optimization
      clf_name: name of regression model. [bayesian_ridge, lasso, elastic_net, ridge, kernel_ridge, svm]
      kernel_func: [linear, rbf, poly]
      degree: degree of polynomial kernel
    example:
      best, tpe_trials = run_hyperopt(train_df, target, 20, 'kernel_ridge', kernel_func='poly')
    """
    def bayesian_opt(params):
        random_seed = 1234
#         n_splits = 3

#         folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

        score_cv = []

        # for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):

        #     y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        #     X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
        X_train, X_valid, y_train, y_valid = X_train_, X_valid_, y_train_, y_valid_

        if clf_name == 'bayesian_ridge':
        
            reg = BayesianRidge(
                n_iter=300, tol=0.001, compute_score=False,
                alpha_1=params['alpha1'], alpha_2=params['alpha2'], 
                lambda_1=params['lambda1'], lambda_2=params['lambda2'],  
                fit_intercept=True, normalize=std_norm, verbose=False)
        
        elif clf_name is 'lasso':
            reg = Lasso(
                alpha=params['alpha'], fit_intercept=True, normalize=std_norm, 
                positive=False, random_state=1234, selection='cyclic')
        
        elif clf_name is 'elastic_net':
            reg = ElasticNet(
                alpha=params['alpha'], l1_ratio=0.5, fit_intercept=True, 
                normalize=std_norm, random_state=1234, selection='cyclic')
        
        elif clf_name is 'ridge':
            reg = Ridge(
                alpha=params['alpha'], fit_intercept=True, normalize=std_norm, 
                max_iter=500, random_state=1234)

        elif clf_name is 'kernel_ridge':
            reg = KernelRidge(
                alpha=params['alpha'], kernel=kernel_func, gamma=None, 
                degree=degree, coef0=1, kernel_params=None)
        
        elif clf_name is 'svm':
            reg = SVR(
                kernel=kernel_func, degree=degree, coef0=0.0, tol=0.001, 
                C=params['alpha'], epsilon=0.1, shrinking=True, cache_size=200, 
                verbose=0, max_iter=500)
        
        else:
            raise Exception('only [bayesian_ridge, lasso, elastic_net, ridge, kernel_ridge, svm] are supported')
        reg.fit(X_train, y_train)
#         score = reg.score(X_valid, y_valid)
        
        pred = reg.predict(X_valid)
        score = mean_squared_error(y_val, pred)
#         score = roc_auc_score(y_valid, oof)
        score_cv.append(score)

        return np.mean(score_cv)

    if clf_name == 'bayesian_ridge':
        bayesian_params = {'alpha1': hp.uniform('alpha1', 0.0001, 5),
                           'alpha2': hp.uniform('alpha2', 0.0001, 5),
                           'lambda1': hp.uniform('lambda1', 0.0001, 5),
                           'lambda2': hp.uniform('lambda2', 0.0001, 5),}   
    else:
        bayesian_params = {'alpha': hp.uniform('alpha', 0.01, 1000),}   

    
    trials = Trials()
    results = fmin(bayesian_opt, bayesian_params, algo=tpe.suggest, 
                   trials=trials, max_evals=max_evals)
        
    return results, trials 


def run_hyperopt_reg(train_df, target, max_evals, clf_name, kernel_func='linear', degree=3):
    """ Return best hyperparameter (mainly regularization parameters)
    Args:
      train_df, target
      max_evals: Total number of iteration to perform for bayesian optimization
      clf_name: name of regression model. [bayesian_ridge, lasso, elastic_net, ridge, kernel_ridge, svm]
      kernel_func: [linear, rbf, poly]
      degree: degree of polynomial kernel
    example:
      best, tpe_trials = run_hyperopt(train_df, target, 20, 'kernel_ridge', kernel_func='poly')
    """
    def bayesian_opt(params):
        random_seed = 1234
        n_splits = 3

        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        oof_lgb = np.zeros(len(train_df))

        score_cv = []

        for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):

            y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
            X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]

            if clf_name == 'bayesian_ridge':
            
                reg = BayesianRidge(
                    n_iter=300, tol=0.001, compute_score=False,
                    alpha_1=params['alpha1'], alpha_2=params['alpha2'], 
                    lambda_1=params['lambda1'], lambda_2=params['lambda2'],  
                    fit_intercept=True, normalize=True, verbose=False)
            
            elif clf_name is 'lasso':
                reg = Lasso(
                    alpha=params['alpha'], fit_intercept=True, normalize=True, 
                    positive=False, random_state=1234, selection='cyclic')
            
            elif clf_name is 'elastic_net':
                reg = ElasticNet(
                    alpha=params['alpha'], l1_ratio=0.5, fit_intercept=True, 
                    normalize=True, random_state=1234, selection='cyclic')
            
            elif clf_name is 'ridge':
                reg = Ridge(
                    alpha=params['alpha'], fit_intercept=True, normalize=True, 
                    max_iter=500, random_state=1234)

            elif clf_name is 'kernel_ridge':
                reg = KernelRidge(
                    alpha=params['alpha'], kernel=kernel_func, gamma=None, 
                    degree=degree, coef0=1, kernel_params=None)
            
            elif clf_name is 'svm':
                reg = SVR(
                    kernel=kernel_func, degree=degree, coef0=0.0, tol=0.001, 
                    C=params['alpha'], epsilon=0.1, shrinking=True, cache_size=200, 
                    verbose=0, max_iter=500)
            
            else:
                raise Exception('only [bayesian_ridge, lasso, elastic_net, ridge, kernel_ridge, svm] are supported')
            reg.fit(X_train, y_train)
            score = reg.score(X_valid, y_valid)
            
    #         score = roc_auc_score(y_valid, oof)
            score_cv.append(score)

        return -np.mean(score_cv)

    if clf_name == 'bayesian_ridge':
        bayesian_params = {'alpha1': hp.uniform('alpha1', 0.0001, 10),
                           'alpha2': hp.uniform('alpha2', 0.0001, 10),
                           'lambda1': hp.uniform('lambda1', 0.0001, 10),
                           'lambda2': hp.uniform('lambda2', 0.0001, 10),}   
    else:
        bayesian_params = {'alpha': hp.uniform('alpha', 0.01, 1000),}   

    
    trials = Trials()
    results = fmin(bayesian_opt, bayesian_params, algo=tpe.suggest, 
                   trials=trials, max_evals=max_evals)
        
    return results, trials 








print("bayesian ridge","="*50)
best, _ = run_hyperopt_reg(train_, valid_, y_tr, y_val, 20, 'bayesian_ridge', std_norm=False, kernel_func='linear', degree=3)
print(best)

print("lasso","="*50)
best, _ = run_hyperopt_reg(train_, valid_, y_tr, y_val, 20, 'lasso', std_norm=False, kernel_func='linear', degree=3)
print(best)

print("elastic_net", "="*50)
best, _ = run_hyperopt_reg(train_, valid_, y_tr, y_val, 20, 'elastic_net', std_norm=False, kernel_func='linear', degree=3)
print(best)

print("ridge", "="*50)
best, _ = run_hyperopt_reg(train_, valid_, y_tr, y_val, 20, 'ridge', std_norm=False, kernel_func='linear', degree=3)
print(best)

print("kernel_ridge", "="*50)
best, _ = run_hyperopt_reg(train_, valid_, y_tr, y_val, 20, 'kernel_ridge', std_norm=False, kernel_func='linear', degree=3)
print(best)

print("kernel_ridge", "="*50)
best, _ = run_hyperopt_reg(train_, valid_, y_tr, y_val, 20, 'kernel_ridge', std_norm=False, kernel_func='rbf', degree=3)
print(best)

print("kernel_ridge", "="*50)
best, _ = run_hyperopt_reg(train_, valid_, y_tr, y_val, 20, 'kernel_ridge', std_norm=False, kernel_func='poly', degree=3)
print(best)

print("svm", "="*50)
best, _ = run_hyperopt_reg(train_, valid_, y_tr, y_val, 20, 'svm', std_norm=False, kernel_func='poly', degree=3)
print(best)
print("="*50)

