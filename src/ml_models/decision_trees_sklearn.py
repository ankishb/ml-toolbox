



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











from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion=criteria, 
  max_depth=depth, max_features=max_features, 
  min_samples_leaf=1, min_samples_split=2,
  random_state=1234, max_leaf_nodes=None, 
  class_weight='balanced')


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(criterion=criteria, 
  max_depth=depth, max_features=max_features,  
  min_samples_leaf=1, min_samples_split=2, 
  random_state=1234, max_leaf_nodes=None)





from sklearn.ensemble import AdaBoostClassifier
AdaBoostClassifier(base_estimator=None, 
  n_estimators=n_estimators, 
  learning_rate=0.1, random_state=1234)


from sklearn.ensemble import AdaBoostRegressor
# loss_func = {‘linear’, ‘square’, ‘exponential’}
AdaBoostRegressor(base_estimator=None, 
  n_estimators=n_estimators, loss=loss_func, 
  learning_rate=0.1, random_state=1234)



"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)

ada_discrete = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME")
ada_discrete.fit(X_train, y_train)

ada_real = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME.R")
ada_real.fit(X_train, y_train)
"""



from sklearn.ensemble import BaggingClassifier
# bootstrap_features: Whether features are drawn with replacement.

BaggingClassifier(base_estimator=None, 
  n_estimators=n_estimators,  
  max_samples=1.0, max_features=1.0, bootstrap=True, 
  bootstrap_features=False, oob_score=False, 
  n_jobs=-1, random_state=1234, verbose=1)

from sklearn.ensemble import BaggingRegressor
BaggingRegressor(base_estimator=None, 
  n_estimators=n_estimators,  
  max_samples=1.0, max_features=1.0, bootstrap=True, 
  bootstrap_features=False, oob_score=False, 
  n_jobs=-1, random_state=1234, verbose=1)



from sklearn.ensemble import ExtraTreesClassifier
"""When looking for the best split to separate the samples 
of a node into two groups, random splits are drawn for each 
of the max_features randomly selected features and the best 
split among those is chosen"""

criteria: ['mae','mse']
etc = ExtraTreesClassifier( 
  criterion=criteria, max_depth=depth, 
  max_features=max_features, n_estimators=n_estimators,
  min_samples_split=2, min_samples_leaf=1, n_jobs=-1,
  min_weight_fraction_leaf=0.0, bootstrap=True, 
  max_leaf_nodes=None, oob_score=False, verbose=1,
  random_state=1234, class_weight='balanced')

from sklearn.ensemble import ExtraTreesRegressor
ExtraTreesRegressor( 
  criterion=criteria, max_depth=depth, 
  max_features=max_features, n_estimators=n_estimators,
  min_samples_split=2, min_samples_leaf=1, n_jobs=-1, 
  min_weight_fraction_leaf=0.0, bootstrap=False,  
  max_leaf_nodes=None, oob_score=False, verbose=1,
  random_state=1234)


def tree():
  """
  Args:
    n_estimators: no of estimators
    criteria: ['gini': gini-impurity,'entropy':infomation gain]
    depth: depth of tree
    min_samples_split : (default=2)
    min_samples_leaf : create smoothening effect
    max_features : ['auto','sqrt','log2',None:n_features]
    max_leaf_nodes : Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
    bootstrap: True
    class_weight : dict, “balanced”, “balanced_subsample” or None
      The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.
  """

from sklearn.tree import ExtraTreeRegressor
criteria = ['mae','mse']
etr = ExtraTreeRegressor(criterion=criteria, max_depth=depth, min_samples_split=2, min_samples_leaf=1, max_features=max_features, random_state=1234, max_leaf_nodes=None)



from sklearn.ensemble import GradientBoostingClassifier
# loss : {‘deviance’, ‘exponential’} (default=’deviance’)
# criteria: Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error
gbc = GradientBoostingClassifier(
  max_features=max_features, n_estimators=n_estimators, 
  subsample=subsample, max_depth=depth, loss=loss_func,
  criterion=criteria, min_samples_split=2, verbose=1, 
  min_samples_leaf=1, random_state=1234, 
  learning_rate=0.1, max_leaf_nodes=None, 
  validation_fraction=0.1, n_iter_no_change=5)


from sklearn.ensemble import GradientBoostingRegressor
# loss_func: {‘ls’, ‘lad’, ‘huber’, ‘quantile’} (default='ls')
# criteria: Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error
# alpha: The alpha-quantile of the huber loss function and the quantile loss function. Only if loss='huber' or loss='quantile'
gbr = GradientBoostingRegressor( 
   max_features=max_features, n_estimators=n_estimators, 
  subsample=subsample, max_depth=depth, loss=loss_func,
  criterion=criteria, min_samples_split=2, verbose=1,
  min_samples_leaf=1, random_state=1234, 
  learning_rate=0.1, alpha=0.9, max_leaf_nodes=None, 
  validation_fraction=0.1, n_iter_no_change=5)


from sklearn.ensemble import RandomForestClassifier
# criteria: ['gini','entropy']
rfc = RandomForestClassifier(n_estimators=n_estimators, 
  criterion=criteria, max_depth=depth, max_features=max_features,  
  min_samples_leaf=1, n_jobs=-1, min_samples_split=2,
  bootstrap=True, max_leaf_nodes=None, oob_score=False, 
  verbose=1, random_state=1234, class_weight='balanced_subsample')

from sklearn.ensemble import RandomForestRegressor
# criteria: ['mae','mse']
rfr = RandomForestRegressor(n_estimators=n_estimators, 
  criterion=criteria, max_depth=depth, max_features=max_features, 
  min_samples_leaf=1, n_jobs=-1, min_samples_split=2,  
  bootstrap=True, max_leaf_nodes=None, oob_score=False,
  verbose=1, random_state=1234)


from sklearn.ensemble import VotingClassifier

voting_criteria = ['hard','soft']
flatten_transform: bool
    Affects shape of transform output only when voting='soft'.
    If voting='soft' and flatten_transform=True, transform method 
    returns matrix with shape (n_samples, n_classifiers * n_classes). 
    If flatten_transform=False, it returns (n_classifiers, n_samples, n_classes).

vote_cf = VotingClassifier(estimators, 
  voting=voting_criteria, 
  weights=weights, n_jobs=-1, flatten_transform=None)

"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                          random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(X, y)
print(eclf1.predict(X))

np.array_equal(eclf1.named_estimators_.lr.predict(X),
               eclf1.named_estimators_['lr'].predict(X))

eclf2 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        voting='soft')
eclf2 = eclf2.fit(X, y)
print(eclf2.predict(X))

eclf3 = VotingClassifier(estimators=[
       ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
       voting='soft', weights=[2,1,1],
       flatten_transform=True)
eclf3 = eclf3.fit(X, y)
print(eclf3.predict(X))

print(eclf3.transform(X).shape)
"""





from sklearn.cluster import KMeans
# precompute_distances: 'auto' 
#   Do not precompute distances if n_samples * n_clusters > 12 million. This corresponds to about 100MB overhead per job using double precision.
kmeans = KMeans(n_clusters=n_clusters, n_init=10, 
  precompute_distances='auto', verbose=1, 
  random_state=1234, n_jobs=-1)

# from sklearn.cluster import KMeans
# import numpy as np
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# kmeans.labels_

# kmeans.predict([[0, 0], [12, 3]])

# kmeans.cluster_centers_



sklearn.cluster import MiniBatchKMeans
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans
mini_kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
  max_iter=200, batch_size=200, verbose=1, 
  compute_labels=True, random_state=1234, 
  max_no_improvement=10, n_init=3, 
  reassignment_ratio=0.01)

"""
from sklearn.cluster import MiniBatchKMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 0], [4, 4],
              [4, 5], [0, 1], [2, 2],
              [3, 2], [5, 5], [1, -1]])
# manually fit on batches
kmeans = MiniBatchKMeans(n_clusters=2,
        random_state=0,
        batch_size=6)
kmeans = kmeans.partial_fit(X[0:6,:])
kmeans = kmeans.partial_fit(X[6:12,:])
kmeans.cluster_centers_


kmeans.predict([[0, 0], [4, 4]])

# fit on the whole data
kmeans = MiniBatchKMeans(n_clusters=2,
        random_state=0,
        batch_size=6,
        max_iter=10).fit(X)
kmeans.cluster_centers_


kmeans.predict([[0, 0], [4, 4]])
"""










