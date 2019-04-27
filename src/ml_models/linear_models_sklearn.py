



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
lin_reg = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0, fit_intercept=True, normalize=False, max_iter=None, random_state=1234)
# If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered)


sklearn.linear_model import RidgeClassifier
ridge_clf = RidgeClassifier(alpha=1.0, fit_intercept=True, 
  normalize=False, class_weight='balanced', random_state=1234)





sklearn.kernel_ridge import KernelRidge
kenel_func = ['rbf','linear','poly']
kernel_ridge = KernelRidge(alpha=1, kernel=kenel_func, gamma=None, 
                          degree=3, coef0=1, kernel_params=None)

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
lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False, 
  positive=False, random_state=1234, selection='cyclic')

"""
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
print(clf.coef_)
print(clf.intercept_) 
"""





from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, 
  normalize=False, random_state=1234, selection='cyclic')
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
bayesian_ridge = BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, 
  lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, verbose=True)
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
logistic_reg = LogisticRegression(penalty=’l2’, dual=False, C=1.0, fit_intercept=True, 
  intercept_scaling=1, class_weight=None, random_state=1234, max_iter=100, 
  multi_class=’warn’, verbose=1, n_jobs=-1)
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
passive_aggresive = PassiveAggressiveClassifier(C=1.0, 
  fit_intercept=True, max_iter=None, tol=None, 
  early_stopping=False, validation_fraction=0.1, 
  n_iter_no_change=5, shuffle=True, verbose=0, 
  n_jobs=-1, random_state=1234, loss='hinge',
  class_weight=None, average=False, n_iter=None)

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
svm = SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)
The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.




from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(loss=’squared_loss’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate=’invscaling’, eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False, n_iter=None)
penalty : str, ‘none’, ‘l2’, ‘l1’, or ‘elasticnet’
learning_rate : string, optional

    The learning rate schedule:

    ‘constant’:

        eta = eta0
    ‘optimal’:

        eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
    ‘invscaling’: [default]

        eta = eta0 / pow(t, power_t)
    ‘adaptive’:

        eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5.

eta0 : double

    The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.0 as eta0 is not used by the default schedule ‘optimal’.
power_t : double

    The exponent for inverse scaling learning rate [default 0.5].
early_stopping : bool, default=False

    Whether to use early stopping to terminate training when validation score is not improving. If set to True, it will automatically set aside a fraction of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.

    New in version 0.20.
validation_fraction : float, default=0.1

    The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True.

    New in version 0.20.
n_iter_no_change : int, default=5

    Number of iterations with no improvement to wait before early stopping.



import numpy as np
from sklearn import linear_model
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
clf.fit(X, y)



from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False, n_iter=None)


The possible options are ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.

n_jobs : int or None, optional (default=None)

    The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
random_state : int, RandomState instance or None, optional (default=None)

    The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
learning_rate : string, optional

    The learning rate schedule:

    ‘constant’:

        eta = eta0
    ‘optimal’: [default]

        eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
    ‘invscaling’:

        eta = eta0 / pow(t, power_t)
    ‘adaptive’:

        eta = eta0, as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5.

eta0 : double

    The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.0 as eta0 is not used by the default schedule ‘optimal’.
power_t : double

    The exponent for inverse scaling learning rate [default 0.5].
early_stopping : bool, default=False

    Whether to use early stopping to terminate training when validation score is not improving. If set to True, it will automatically set aside a fraction of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.

    New in version 0.20.
validation_fraction : float, default=0.1

    The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True.

    New in version 0.20.
n_iter_no_change : int, default=5

    Number of iterations with no improvement to wait before early stopping.



return predict_proba
   





clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))



# Fitting a simple Logistic Regression on Counts
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, ytrain)
predictions = clf.predict_proba(xvalid_ctv)




#Base models

#    LASSO Regression :
#This model may be very sensitive to outliers. So we need to made it more robust on them. 
#For that we use the sklearn's Robustscaler() method on pipeline

# lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso = Lasso(alpha =0.0005, random_state=1)

#    Elastic Net Regression :
#again made robust to outliers

# ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

#    Kernel Ridge Regression :
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

#    Gradient Boosting Regression :
#With huber loss that makes it robust to outliers
GBoost = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,#n_estimators=3000
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

#    XGBoost :
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=220,#n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

#    LightGBM :
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=220,#n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

