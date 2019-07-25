



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb



from sklearn.linear_model import LinearRegression
lin_reg_params = fit_intercept=True, normalize=False, n_jobs=-1
lin_reg = LinearRegression(lin_reg_params)


from sklearn.linear_model import Ridge
ridge_params = alpha=1.0, fit_intercept=True, normalize=False, max_iter=None, random_state=1234
ridge = Ridge(ridge_params)

# If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered)


sklearn.linear_model import RidgeClassifier
ridge_clf_params = alpha=1.0, fit_intercept=True, normalize=False, class_weight='balanced', random_state=1234
ridge_clf = RidgeClassifier(ridge_clf_params)





sklearn.kernel_ridge import KernelRidge
kenel_func = ['rbf','linear','poly']
kernel_ridge_params = alpha=1, kernel=kenel_func, gamma=None,  degree=3, coef0=1, kernel_params=None
kernel_ridge = KernelRidge(kernel_ridge_params)

# gamma : Gamma parameter for the RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels. 
#         Interpretation of the default value is left to the kernel

"""
from sklearn.kernel_ridge import KernelRidge
import numpy as np
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = KernelRidge(alpha=1.0)
clf.fit(X, y)
""" 


from sklearn.linear_model import Lasso
lasso_params = alpha=1.0, fit_intercept=True, normalize=False, positive=False, random_state=1234, selection='cyclic'
lasso = Lasso(lasso_params)

"""
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
print(clf.coef_)
print(clf.intercept_) 
"""





from sklearn.linear_model import ElasticNet
elastic_params = alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, random_state=1234, selection='cyclic'
elastic_net = ElasticNet(elastic_params)
  For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty.

"""
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression

X, y = make_regression(n_features=2, random_state=0)
regr = ElasticNet(random_state=0)
regr.fit(X, y)

print(regr.coef_) 
print(regr.intercept_) 
print(regr.predict([[0, 0]])
"""




from sklearn.linear_model import BayesianRidge
bayesian_ridge_params = n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, verbose=True
bayesian_ridge = BayesianRidge(bayesian_ridge_params)
# alpha_1 : Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter. Default is 1.e-6
# alpha_2 : Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter. Default is 1.e-6
# lambda_1 : Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter. Default is 1.e-6.
# lambda_2 : Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter. Default is 1.e-6

"""
from sklearn import linear_model
clf = linear_model.BayesianRidge()
clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
clf.predict([[1, 1]])
"""



from sklearn.linear_model import LogisticRegression
logistic_reg_params = penalty=’l2’, dual=False, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=1234, max_iter=100,  multi_class=’warn’, verbose=1, n_jobs=-1
logistic_reg = LogisticRegression(logistic_reg_params)

multi_class : str, {‘ovr’, ‘multinomial’, ‘auto’}, default: ‘ovr’
If the option chosen is ‘ovr’, then a binary problem is fit for each label.
For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.


"""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :]) 
clf.score(X, y)
"""



from sklearn.linear_model import PassiveAggressiveClassifier
passive_aggresive_params = C=1.0, fit_intercept=True, max_iter=None, tol=None, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0, n_jobs=-1, random_state=1234, loss='hinge', class_weight=None, average=False, n_iter=None
passive_aggresive = PassiveAggressiveClassifier(passive_aggresive_params)

# early_stopping=True
# n_iter_no_change : Number of iterations with no improvement to wait before early stopping.


"""
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3)
clf.fit(X, y)
print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[0, 0, 0, 0]]))
"""

The index t has been chosen to mark the temporal dimension. In this case, in fact, the samples can continue arriving for an indefinite time. Of course, if they are drawn from same data generating distribution, the algorithm will keep learning (probably without large parameter modifications), but if they are drawn from a completely different distribution, the weights will slowly forget the previous one and learn the new distribution.





from sklearn.svm import SVC
svm_params = C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None
svm = SVC(svm_params)

The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.




from sklearn.linear_model import SGDRegressor
sgd_reg_params = loss=’squared_loss’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate=’invscaling’, eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False, n_iter=None
sgd_reg = SGDRegressor(sgd_reg_params)

penalty : str, ‘none’, ‘l2’, ‘l1’, or ‘elasticnet’
learning_rate : The learning rate schedule:
    ‘constant’: eta = eta0
    ‘optimal’: eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
    ‘invscaling’: [default] eta = eta0 / pow(t, power_t)
    ‘adaptive’: eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5.

eta0 :  The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.0 as eta0 is not used by the default schedule ‘optimal’.
power_t : The exponent for inverse scaling learning rate [default 0.5].
early_stopping : bool, default=False, 
  Whether to use early stopping to terminate training when validation score is not improving. If set to True, it will automatically set aside a fraction of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
validation_fraction : float, default=0.1, Only used if early_stopping is True.
n_iter_no_change : int, default=5



import numpy as np
from sklearn import linear_model
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
clf.fit(X, y)



from sklearn.linear_model import SGDClassifier
sgd_clf_params = loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False, n_iter=None
sgd_clf = SGDClassifier(sgd_clf_params)



The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.

random_state : int, RandomState instance or None, optional (default=None)
learning_rate :  The learning rate schedule:

    ‘constant’: eta = eta0
    ‘optimal’: [default] eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
    ‘invscaling’: eta = eta0 / pow(t, power_t)
    ‘adaptive’: eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5.

eta0 : The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.0 as eta0 is not used by the default schedule ‘optimal’.
power_t :  The exponent for inverse scaling learning rate [default 0.5].
early_stopping : bool, default=False
  Whether to use early stopping to terminate training when validation score is not improving. If set to True, it will automatically set aside a fraction of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
validation_fraction : float, default=0.1, Only used if early_stopping is True.
n_iter_no_change : int, default=5








import numpy as np
import pandas as pd
from sklearn.datasets import load_boston


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR


boston = load_boston()

X = boston['data']
y = boston['target']

cols = list(boston.feature_names) + ['target']

df = pd.DataFrame(data=np.column_stack([X,y]), columns=cols)

idx = np.random.permutation(df.shape[0])

train_len = int(len(idx)*0.8)
train_df = df.iloc[idx][:train_len]
test_df  = df.iloc[idx][train_len:]

train_df.shape, test_df.shape


bayesian_ridge = BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, 
                               lambda_1=1e-06, lambda_2=1e-06, compute_score=False, 
                               fit_intercept=True, normalize=True, verbose=False)

lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=True, positive=False, 
              random_state=1234, selection='cyclic')

elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=True, 
                         random_state=1234, selection='cyclic')
#   For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty.

lin_reg = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)

ridge = Ridge(alpha=1.0, fit_intercept=True, normalize=True, max_iter=500, 
              random_state=1234)

# kenel_func = ['rbf','linear','poly']
kernel_ridge = KernelRidge(alpha=1, kernel='rbf', gamma=None, degree=3, coef0=1, 
                           kernel_params=None)

svm = SVR(kernel='rbf', degree=3, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, 
          shrinking=True, cache_size=200, verbose=False, max_iter=500)

bayesian_ridge.fit(train_df.drop('target', axis=1), train_df['target'])
print(bayesian_ridge.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")

lasso.fit(train_df.drop('target', axis=1), train_df['target'])
print(lasso.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")

elastic_net.fit(train_df.drop('target', axis=1), train_df['target'])
print(elastic_net.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")

lin_reg.fit(train_df.drop('target', axis=1), train_df['target'])
print(lin_reg.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")

ridge.fit(train_df.drop('target', axis=1), train_df['target'])
print(ridge.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")

kernel_ridge.fit(train_df.drop('target', axis=1), train_df['target'])
print(kernel_ridge.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")

svm.fit(train_df.drop('target', axis=1), train_df['target'])
print(svm.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")





from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC

passive_agg = PassiveAggressiveClassifier(C=params['alpha'], fit_intercept=True, max_iter=None, tol=None, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0, n_jobs=-1, random_state=1234, loss='hinge', class_weight='balanced', average=False, n_iter=None)


ridge_clf = RidgeClassifier(alpha=params['alpha'], fit_intercept=True, normalize=True, class_weight='balanced', random_state=1234)

svm = SVC(C=1.0, kernel='rbf', degree=3, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', random_state=None)

logistic_reg = LogisticRegression(penalty='l2', dual=False, C=params['alpha'], fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=1234, max_iter=100, multi_class='warn', verbose=0, n_jobs=-1)




from sklearn.datasets import load_iris
iris = load_iris()

X = iris['data']
y = iris['target']

cols = list(iris.feature_names) + ['target']

df = pd.DataFrame(data=np.column_stack([X,y]), columns=cols)

idx = np.random.permutation(df.shape[0])

train_len = int(len(idx)*0.8)
train_df = df.iloc[idx][:train_len]
test_df  = df.iloc[idx][train_len:]

train_df.shape, test_df.shape




ridge_clf.fit(train_df.drop('target', axis=1), train_df['target'])
print(ridge_clf.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")

passive_agg.fit(train_df.drop('target', axis=1), train_df['target'])
print(passive_agg.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")

svm.fit(train_df.drop('target', axis=1), train_df['target'])
print(svm.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")

logistic_reg.fit(train_df.drop('target', axis=1), train_df['target'])
print(logistic_reg.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")

ridge_clf.fit(train_df.drop('target', axis=1), train_df['target'])
print(ridge_clf.score(test_df.drop('target', axis=1), test_df['target']))
print("==============")




def run_hyperopt_clf(train_df, target, max_evals, clf_name, kernel_func='linear', degree=3):

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


best, tpe_trials = run_hyperopt_clf(train_df, target, 20, 'passive_agg')
print(best)
print("="*50)

best, tpe_trials = run_hyperopt_clf(train_df, target, 100, 'ridge')
print(best)
print("="*50)

best, tpe_trials = run_hyperopt_clf(train_df, target, 100, 'logistic')
print(best)
print("="*50)

best, tpe_trials = run_hyperopt_clf(train_df, target, 20, 'svm')
print(best)
print("="*50)

best, tpe_trials = run_hyperopt_clf(train_df, target, 20, 'svm', kernel_func='rbf')
print(best)
print("="*50)

best, tpe_trials = run_hyperopt_clf(train_df, target, 20, 'svm', kernel_func='poly')
print(best)
print("="*50)






def run_hyperopt_reg(train_df, target, max_evals, clf_name, kernel_func='linear', degree=3):

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
                raise Exception('only [bayesian_ridge, lasso, elastic_net, ridge, kernel_ridge, svm] \
                are supported')
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
best, tpe_trials = run_hyperopt(train_df, target, 20, 'bayesian_ridge')
print(best)

print("lasso","="*50)
best, tpe_trials = run_hyperopt(train_df, target, 20, 'lasso')
print(best)

print("elastic_net", "="*50)
best, tpe_trials = run_hyperopt(train_df, target, 100, 'elastic_net')
print(best)

print("ridge", "="*50)
best, tpe_trials = run_hyperopt(train_df, target, 100, 'ridge')
print(best)

print("kernel_ridge", "="*50)
best, tpe_trials = run_hyperopt(train_df, target, 20, 'kernel_ridge')
print(best)

print("kernel_ridge", "="*50)
best, tpe_trials = run_hyperopt(train_df, target, 20, 'kernel_ridge', kernel_func='rbf')
print(best)

print("kernel_ridge", "="*50)
best, tpe_trials = run_hyperopt(train_df, target, 20, 'kernel_ridge', kernel_func='poly')
print(best)

print("svm", "="*50)
best, tpe_trials = run_hyperopt(train_df, target, 20, 'svm', kernel_func='poly')
print(best)
print("="*50)