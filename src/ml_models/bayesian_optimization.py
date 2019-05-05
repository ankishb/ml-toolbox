
def baysian_opt():
	"""
	To learn a function, we pick one sample and fit an approximated all possible function passing throgh that point, this is represented by the covariance map with an interval of 95%, this will gives us the idea where to look next.

	To learn more quickly, we will explore the area with highly covaraince, with more sample from that place, will reduce the uncertainty. This is achieved by the acquisition function.

	1. We have observation, let's say we have 2 data sample {(x1,y1),(x2,y2)}.
	2. Maximize the score by finding the best parameters using acquisition function.
	3. For this parameter, compute the prediction y as f(x)+epsilon (using gp), (Now we have 3 points)
	4. Update GP and repeat step 1-4.
	Ref:
		https://www.youtube.com/watch?v=vz3D36VXefI&index=10&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6
		https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation_vs_exploration.ipynb
	
	Acquisition Function "Upper Confidence Bound"
		Prefer exploitation (kappa=1.0)
		Prefer exploration (kappa=10.0)
	Acquisition Function "Expected Improvement"
		Prefer exploitation (xi=1e-4)
		Prefer exploration (xi=1e-1)
	Acquisition Function "Probability of Improvement"
		Prefer exploitation (xi=1e-4)
		Prefer Exploration (xi=1e-1)

	
	"""
	pass


def bayesian_blending(x1,x2,x3,x4,x5,x6,x7,x8):
    blend_arr = np.array([x1,x2,x3,x4,x5,x6,x7,x8,0,0,0,0])
    oof = np.dot(oof_all.values, blend_arr)
    score = 100*np.sqrt(mean_squared_error(y_val, oof))
    
    return -score

params = {
    'x1'    : (0.1,0.7),
    'x2'    : (0.1,0.7),
    'x3'    : (0.1,0.7),
    'x4'    : (0.1,0.7),
    'x5'    : (0.1,0.7),
    'x6'    : (0.1,0.7),
    'x7'    : (0.1,0.7),
    'x8'    : (0.1,0.7),
#     'x9'    : (0,1),
#     'x10'   : (0,1),
#     'x11'   : (0,1),
#     'x12'   : (0,1),
    }
_bo = BayesianOptimization(bayesian_blending, params, random_state=26656)
_bo.maximize(init_points=100, n_iter=50, acq='ei')



from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

def bayesian_opt_cat(X_train, y_train, X_valid, y_valid, features):
        
    def train_cat_model(r_str, b_temp, l2, depth):
    
        params = {}

        params['random_strength']     = max(min(r_str, 1), 0)
        params['bagging_temperature'] = max(b_temp, 0)
        
        params['l2_leaf_reg'] = max(l2, 0)
#         params['min_data_in_leaf'] = max(data_leaf,1)
        
#         params['subsample'] = subsample
        params['depth']     = int(depth)

        param_const = {
            'border_count'          : 63,
            'early_stopping_rounds' : 50,
            'random_seed'           : 1337,
            'task_type'             : 'CPU', 
            'loss_function'         : "RMSE", 
    #         'subsample'             = 0.7, 
            'iterations'            : 10000, 
            'learning_rate'         : 0.01,
            'thread_count'          : 4,
#             'bootstrap_type'        : 'No'
        }

        for key, item in param_const.items():
            params[key] = item
    
        

        _train = Pool(X_train[features], label=y_train)#, cat_features=cate_features_index)
        _valid = Pool(X_valid[features], label=y_valid)#, cat_features=cate_features_index)

        watchlist = [_train, _valid]
        clf = CatBoostRegressor(**params)
        clf.fit(_train, 
                eval_set=watchlist, 
                verbose=0,
                use_best_model=True)

        oof  = clf.predict(X_valid[features])

        score = mean_squared_error(y_valid, oof)

        return -score

    _bo = BayesianOptimization(train_cat_model, {

        'r_str'      : (1, 5),
        'b_temp'     : (0.01, 100),
        'depth'      : (3,8), # int
#         'subsample'  : (0.3, 0.8),
#         'data_leaf'  : (2,7),
        'l2'         : (0, 5),

    }, random_state=23456)
    _bo.maximize(init_points=25, n_iter=12, acq='ei')
    
    return _bo

####################################################################################################################################################################################################################################################
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

def bayesian_opt_lgb(X_train, y_train, X_valid, y_valid, features):
        
    def train_lgb_model(f_frac, b_frac, 
                        l1, l2, split_gain,
                        leaves, data_in_leaf, hessian):
    
        param = {}

        param['feature_fraction'] = max(min(f_frac, 1), 0)
        param['bagging_fraction'] = max(min(b_frac, 1), 0)

        param['lambda_l1'] = max(l1, 0)
        param['lambda_l2'] = max(l2, 0)
        param['min_split_gain'] = split_gain
#     #     params['min_child_weight'] = min_child_weight

        param['num_leaves'] = int(leaves)
        param['min_data_in_leaf'] = int(data_in_leaf)
        param['min_sum_hessian_in_leaf'] = max(hessian, 0)

        param_const = {
            'max_bins'               : 63,
            'learning_rate'          : 0.01,
            'num_threads'            : 4,
            'metric'                 : 'rmse',
            'boost'                  : 'gbdt',
            'tree_learner'           : 'serial',
            'objective'              : 'root_mean_squared_error',
            'verbosity'              : 0,
        }

        for key, item in param_const.items():
            param[key] = item
    
#         print(param)

        _train = lgb.Dataset(X_train[features], label=y_train, feature_name=list(features))
        _valid = lgb.Dataset(X_valid[features], label=y_valid,feature_name=list(features))

        clf = lgb.train(param, _train, 10000, 
                        valid_sets = [_train, _valid], 
                        verbose_eval=0, 
                        early_stopping_rounds = 25)                  

        oof = clf.predict(X_valid[features], num_iteration=clf.best_iteration)
        score = mean_squared_error(y_valid, oof)

        return -score


    _bo = BayesianOptimization(train_lgb_model, {
        # speed
#         'bagging_freq'           : 5, #int
        'b_frac'       : (0.2,0.7),
        'f_frac'       : (0.2,0.8),

#         # accuracy
# #         'max_bins'               : 127,
#         'learning_rate'          : 0.01,
        'leaves'             : (20,90), # int
    
        # regularization
        'split_gain'      : (0, 10),
        'l1'              : (0, 5),
        'l2'              : (0, 5),
        
#         # deal with overfitting
        'data_in_leaf'       : (20, 500), # int
        'hessian': (0, 100),


    }, random_state=23456)
    _bo.maximize(init_points=25, n_iter=12, acq='ei')
    
    return _bo
####################################################################################################################################################################################################################################################





Exploitation vs Exploration
In [1]:

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization

Target function
In [2]:

np.random.seed(42)
xs = np.linspace(-2, 10, 10000)

def f(x):
    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1/ (x ** 2 + 1)

plt.plot(xs, f(xs))
plt.show()

Utility function for plotting
In [10]:

def plot_bo(f, bo):
    x = np.linspace(-2, 10, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)
    
    plt.figure(figsize=(16, 9))
    plt.plot(x, f(x))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.show()

Acquisition Function "Upper Confidence Bound"
Prefer exploitation (kappa=1.0)

Note that most points are around the peak(s).
In [11]:

bo = BayesianOptimization(
    f=f,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10, acq="ucb", kappa=0.1)

plot_bo(f, bo)

Prefer exploration (kappa=10)

Note that the points are more spread out across the whole range.
In [5]:

bo = BayesianOptimization(
    f=f,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10, acq="ucb", kappa=10)

plot_bo(f, bo)

Acquisition Function "Expected Improvement"
Prefer exploitation (xi=0.0)

Note that most points are around the peak(s).
In [6]:

bo = BayesianOptimization(
    f=f,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10, acq="ei", xi=1e-4)

plot_bo(f, bo)

Prefer exploration (xi=0.1)

Note that the points are more spread out across the whole range.
In [7]:

bo = BayesianOptimization(
    f=f,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10, acq="ei", xi=1e-1)

plot_bo(f, bo)

Acquisition Function "Probability of Improvement"
Prefer exploitation (xi=0.0)

Note that most points are around the peak(s).
In [8]:

bo = BayesianOptimization(
    f=f,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10, acq="poi", xi=1e-4)

plot_bo(f, bo)

Prefer exploration (xi=0.1)

Note that the points are more spread out across the whole range.
In [9]:

bo = BayesianOptimization(
    f=f,
    pbounds={"x": (-2, 10)},
    verbose=0,
    random_state=987234,
)

bo.maximize(n_iter=10, acq="poi", xi=1e-1)

plot_bo(f, bo)









def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {'eval_metric': 'rmse',
              'max_depth': int(max_depth),
              'subsample': 0.8,
              'eta': 0.1,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3, 0.9)}
                                             )
# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')

Initialization
----------------------------------------------------------------------------
 Step |   Time |      Value |   colsample_bytree |     gamma |   max_depth | 
    1 | 08m17s |   -3.73421 |             0.6682 |    0.3421 |      6.1668 | 
    2 | 05m59s |   -3.87765 |             0.7762 |    0.7602 |      4.3337 | 
    3 | 02m45s |   -4.06169 |             0.3768 |    0.9385 |      3.3595 | 
Bayesian Optimization
----------------------------------------------------------------------------
 Step |   Time |      Value |   colsample_bytree |     gamma |   max_depth | 
    4 | 13m43s |   -3.68866 |             0.9000 |    1.0000 |      7.0000 | 
    5 | 13m39s |   -3.68688 |             0.9000 |    0.0000 |      7.0000 | 
    6 | 05m07s |   -3.73433 |             0.3000 |    0.0000 |      7.0000 | 
    7 | 11m14s |   -3.73010 |             0.8995 |    0.5228 |      6.9917 | 
    8 | 09m24s |   -3.79521 |             0.9000 |    0.0000 |      5.5249 | 

Extract the parameters of the best model.

params = xgb_bo.res['max']['max_params']
params['max_depth'] = int(params['max_depth'])





bayes_lgb_params = {
        # speed
#         'bagging_freq'           : 5, #int
        'bagging_fraction'       : (0.2,0.7),
        'feature_fraction'       : (0.2,0.8),

        # accuracy
        'max_bins'               : 127,
        'learning_rate'          : 0.01,
        'num_leaves'             : (20,90), # int
    
        # regularization
        'min_gain_to_split'      : (0, 10),
        'lambda_l1'              : (0, 5),
        'lambda_l2'              : (0, 5),
        
        # deal with overfitting
        'min_data_in_leaf'       : (20, 500), # int
        'min_sum_hessian_in_leaf': (0, 100),
        
        'num_threads'            : 4,
        'metric'                 : 'rmse',
        'boost'                  : 'gbdt',
        'tree_learner'           : 'serial',
        'objective'              : 'root_mean_squared_error',
        'verbosity'              : 1,

    }

import lightgbm as lgb


def bayesian_opt_lgb(X_train, y_train, X_valid, y_valid, features, param):
        
    def train_lgb_model(param):
    
#         param['feature_fraction'] = max(min(feature_fraction, 1), 0)
#         param['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

#         param['lambda_l1'] = max(lambda_l1, 0)
#         param['lambda_l2'] = max(lambda_l2, 0)
#         param['min_split_gain'] = min_split_gain
#     #     params['min_child_weight'] = min_child_weight

        param['num_leaves'] = int(param['num_leaves'])
#         param['min_data_in_leaf'] = int(param['min_data_in_leaf'])
#         param['min_sum_hessian_in_leaf'] = max(param['min_sum_hessian_in_leaf'], 0)

        param_const = {
        'num_threads'            : 4,
        'metric'                 : 'rmse',
        'boost'                  : 'gbdt',
        'tree_learner'           : 'serial',
        'objective'              : 'root_mean_squared_error',
        'verbosity'              : 1,}

        for key, item in param_const.items():
            param[key] = item
    
        print(param)

        _train = lgb.Dataset(X_train[features], label=y_train, feature_name=list(features))
        _valid = lgb.Dataset(X_valid[features], label=y_valid,feature_name=list(features))

        clf = lgb.train(param, _train, 10000, 
                        valid_sets = [_train, _valid], 
                        verbose_eval=200, 
                        early_stopping_rounds = 25)                  

        oof = clf.predict(X_valid[features], num_iteration=clf.best_iteration)
        score = mean_squared_error(y_valid, oof)

        return -0.1


    _bo = BayesianOptimization(train_lgb_model, {
        # speed
#         'bagging_freq'           : 5, #int
#         'bagging_fraction'       : (0.2,0.7),
#         'feature_fraction'       : (0.2,0.8),

#         # accuracy
# #         'max_bins'               : 127,
#         'learning_rate'          : 0.01,
        'num_leaves'             : (20,90), # int
    
        # regularization
#         'min_gain_to_split'      : (0, 10),
#         'lambda_l1'              : (0, 5),
#         'lambda_l2'              : (0, 5),
        
#         # deal with overfitting
#         'min_data_in_leaf'       : (20, 500), # int
#         'min_sum_hessian_in_leaf': (0, 100),


    }, random_state=23456)
    _bo.maximize(init_points=1, n_iter=5, acq='ei')
    
    return _bo



################### example how to use for LGBM models ####################



X = application_train.drop('TARGET', axis=1)
y = application_train.TARGET
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, categorical_feature = categorical_feats, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO.res['max']['max_params']

opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=100, learning_rate=0.05)

Initialization
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Step |   Time |      Value |   bagging_fraction |   feature_fraction |   lambda_l1 |   lambda_l2 |   max_depth |   min_child_weight |   min_split_gain |   num_leaves | 
    1 | 00m59s |    0.74761 |             0.9583 |             0.6167 |      4.8931 |      1.9198 |      5.3476 |            32.7936 |           0.0272 |      35.5251 | 
    2 | 00m59s |    0.74885 |             0.9058 |             0.4501 |      3.9958 |      0.4301 |      5.0807 |            32.5443 |           0.0776 |      39.0190 | 
    3 | 01m09s |    0.74990 |             0.9136 |             0.8134 |      2.3074 |      2.8340 |      8.3222 |            32.7620 |           0.0462 |      36.6580 | 
    4 | 00m51s |    0.75006 |             0.9851 |             0.8709 |      3.9026 |      1.5655 |      8.1048 |            47.4687 |           0.0573 |      35.4425 | 
    5 | 00m35s |    0.75089 |             0.8142 |             0.4068 |      0.5914 |      1.2440 |      8.4713 |            35.6819 |           0.0029 |      32.8968 | 
Bayesian Optimization
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Step |   Time |      Value |   bagging_fraction |   feature_fraction |   lambda_l1 |   lambda_l2 |   max_depth |   min_child_weight |   min_split_gain |   num_leaves | 
    6 | 00m45s |    0.75122 |             0.8824 |             0.2785 |      0.0636 |      2.2423 |      5.8154 |            49.1417 |           0.0100 |      44.1038 | 
    7 | 00m58s |    0.75131 |             0.9785 |             0.8560 |      0.2812 |      0.0292 |      8.9339 |            49.9035 |           0.0821 |      43.8205 | 
    8 | 00m55s |    0.75166 |             0.9212 |             0.2732 |      0.2660 |      1.9191 |      8.8121 |             5.3852 |           0.0012 |      44.5129 | 
    9 | 00m47s |    0.73409 |             0.8018 |             0.1068 |      4.9587 |      2.9412 |      8.6253 |            39.9238 |           0.0773 |      44.9553 | 
   10 | 00m46s |    0.74762 |             0.8871 |             0.7749 |      0.3189 |      0.4824 |      5.3296 |             5.3105 |           0.0562 |      25.0965 | 
   11 | 00m45s |    0.74755 |             0.9459 |             0.8495 |      0.8824 |      2.9314 |      5.1176 |            49.9633 |           0.0341 |      24.0223 | 
   12 | 00m49s |    0.74780 |             0.8840 |             0.8662 |      0.3931 |      0.0145 |      5.0739 |             5.3169 |           0.0658 |      41.4870 | 
   13 | 00m54s |    0.74997 |             0.9101 |             0.8600 |      4.9105 |      1.7085 |      8.9306 |             5.3815 |           0.0019 |      32.6167 | 
   14 | 00m52s |    0.74996 |             0.9629 |             0.8443 |      0.7885 |      0.0156 |      8.9880 |            13.5156 |           0.0184 |      32.2845 | 
   15 | 00m48s |    0.74793 |             0.9969 |             0.8076 |      0.6626 |      0.0229 |      5.0210 |            44.9239 |           0.0965 |      36.0874 | 

################### example how to use for LGBM models ####################














################### example how to use for sklearn models ####################

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours

def get_data():
    """Synthetic binary classification dataset."""
    data, targets = make_classification(
        n_samples=1000,
        n_features=45,
        n_informative=12,
        n_redundant=7,
        random_state=134985745,
    )
    return data, targets


def svc_cv(C, gamma, data, targets):
    """SVC cross validation.
    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    estimator = SVC(C=C, gamma=gamma, random_state=2)
    cval = cross_val_score(estimator, data, targets, scoring='roc_auc', cv=4)
    return cval.mean()


def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    estimator = RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=2
    )
    cval = cross_val_score(estimator, data, targets,
                           scoring='neg_log_loss', cv=4)
    return cval.mean()


def optimize_svc(data, targets):
    """Apply Bayesian Optimization to SVC parameters."""
    def svc_crossval(expC, expGamma):
        """Wrapper of SVC cross validation.
        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        C = 10 ** expC
        gamma = 10 ** expGamma
        return svc_cv(C=C, gamma=gamma, data=data, targets=targets)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)


def optimize_rfc(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (10, 250),
            "min_samples_split": (2, 25),
            "max_features": (0.1, 0.999),
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)

if __name__ == "__main__":
    data, targets = get_data()

    print(Colours.yellow("--- Optimizing SVM ---"))
    optimize_svc(data, targets)

    print(Colours.green("--- Optimizing Random Forest ---"))
	optimize_rfc(data, targets)


















from sklearn.metrics import make_scorer

def rmsle(y_true, y_pred):
    # Remember, we transformed price with log1p previously.
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))

neg_rmsle = make_scorer(rmsle, greater_is_better=False)



from bayes_opt import BayesianOptimization
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge

seed = 101 # Lucky seed

    
def target(**params):
    fit_intercept = int(params['fit_intercept'])
    fit_intercept_dict = {0:False, 1:True}

    model = Ridge(alpha = params['alpha'],
                    fit_intercept = fit_intercept_dict[fit_intercept],
                    copy_X = True)
    
    scores = cross_val_score(model, train_X, train_y, scoring=neg_rmsle, cv=3)
    return scores.mean()
    
params = {'alpha':(1, 4),
          'fit_intercept':(0,1.99)}
if develop:
    bo = BayesianOptimization(target, params, random_state=seed)
    bo.gp.set_params(alpha=1e-8)
    bo.maximize(init_points=5, n_iter=10, acq='ucb', kappa=2)
    
    print(bo.res['max']['max_params'])














lgbm_params = {
            'nthread': 4,
            'n_estimators': 10000,
            'learning_rate': .02,
            'num_leaves': 34,
            'colsample_bytree': .9497036,
            'subsample': .8715623,
            'max_depth': 8,
            'reg_alpha': .041545473,
            'reg_lambda': .0735294,
            'min_split_gain': .0222415,
            'min_child_weight': 39.3259775,
            'silent': -1,
            'verbose': -1
}

Bayesian Optimization

def lgbm_evaluate(**params):
    warnings.simplefilter('ignore')
    
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
        
    clf = LGBMClassifier(**params, n_estimators = 10000, nthread = 4)

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    folds = KFold(n_splits = 2, shuffle = True, random_state = 1001)
        
    test_pred_proba = np.zeros(train_df.shape[0])
    
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf.fit(train_x, train_y, 
                eval_set = [(train_x, train_y), (valid_x, valid_y)], eval_metric = 'auc', 
                verbose = False, early_stopping_rounds = 100)

        test_pred_proba[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, 1]
        
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    return roc_auc_score(train_df['TARGET'], test_pred_proba)

params = {'colsample_bytree': (0.8, 1),
          'learning_rate': (.01, .02), 
          'num_leaves': (33, 35), 
          'subsample': (0.8, 1), 
          'max_depth': (7, 9), 
          'reg_alpha': (.03, .05), 
          'reg_lambda': (.06, .08), 
          'min_split_gain': (.01, .03),
          'min_child_weight': (38, 40)}
#bo = BayesianOptimization(lgbm_evaluate, params)
#bo.maximize(init_points = 5, n_iter = 5)

#best_params = bo.res['max']['max_params']
#best_params['num_leaves'] = int(best_params['num_leaves'])
#best_params['max_depth'] = int(best_params['max_depth'])

#best_params

#bo.res['max']['max_val']

