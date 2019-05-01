


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier




import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston


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


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier


def tree(tree_name, criteria, depth, n_estimators, lr_rate=0.1, max_features=0.7, min_samples_leaf=3, max_samples=0.75, subsample=0.7, loss_func='mse'):
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
    if tree_name is 'decision_tree_clf':
        dtc = DecisionTreeClassifier(criterion=criteria, max_depth=depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=2,random_state=1234, max_leaf_nodes=None, class_weight='balanced')

    if tree_name is 'decision_tree_reg':
        dtr = DecisionTreeRegressor(criterion=criteria, max_depth=depth, max_features=max_features,  min_samples_leaf=min_samples_leaf, min_samples_split=2, random_state=1234, max_leaf_nodes=None)


    if tree_name is 'ada_boost_clf':
        ada_noost = AdaBoostClassifier(base_estimator=None, n_estimators=n_estimators, learning_rate=lr_rate, random_state=1234)


    if tree_name is 'ada_boost_reg':
    # loss_func = {‘linear’, ‘square’, ‘exponential’}
        AdaBoostRegressor(base_estimator=None, n_estimators=n_estimators, loss=loss_func, learning_rate=lr_rate, random_state=1234)


    if tree_name is 'bagging_clf':
    # bootstrap_features: Whether features are drawn with replacement.
        bag_clf = BaggingClassifier(base_estimator=None, n_estimators=n_estimators,  max_samples=max_samples, max_features=max_features, bootstrap=True, bootstrap_features=False, oob_score=False, n_jobs=-1, random_state=1234, verbose=1)


    if tree_name is 'bagging_reg':
        bag_reg = BaggingRegressor(base_estimator=None, n_estimators=n_estimators,  max_samples=max_samples, max_features=max_features, bootstrap=True, bootstrap_features=False, oob_score=False, n_jobs=-1, random_state=1234, verbose=1)


    if tree_name is 'extra_tree_clf':
        """When looking for the best split to separate the samples 
        of a node into two groups, random splits are drawn for each 
        of the max_features randomly selected features and the best 
        split among those is chosen"""

#     criteria: ['mae','mse']
        et_clf = ExtraTreesClassifier( criterion=criteria, max_depth=depth, max_features=max_features, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=2, n_jobs=-1, bootstrap=True, max_leaf_nodes=None, oob_score=False, verbose=1,random_state=1234, class_weight='balanced')

    if tree_name is 'extra_tree_reg':
        et_reg = ExtraTreesRegressor( criterion=criteria, max_depth=depth, max_features=max_features, n_estimators=n_estimators,min_samples_leaf=min_samples_leaf, min_samples_split=2, n_jobs=-1, bootstrap=False,  max_leaf_nodes=None, oob_score=False, verbose=1,random_state=1234)



    if tree_name is 'grad_boost_clf':

    # loss : {‘deviance’, ‘exponential’} (default=’deviance’)
    # criteria: Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error
        grad_boost_clf = GradientBoostingClassifier(max_features=max_features, n_estimators=n_estimators, subsample=subsample, max_depth=depth, loss=loss_func, criterion=criteria, min_samples_leaf=min_samples_leaf, learning_rate=lr_rate, min_samples_split=2, verbose=1, random_state=1234, max_leaf_nodes=None, validation_fraction=0.1, n_iter_no_change=5)


    if tree_name is 'grad_boost_reg':

    # loss_func: {‘ls’, ‘lad’, ‘huber’, ‘quantile’} (default='ls')
    # criteria: Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error
    # alpha: The alpha-quantile of the huber loss function and the quantile loss function. Only if loss='huber' or loss='quantile'
        grad_boost_reg = GradientBoostingRegressor( max_features=max_features, n_estimators=n_estimators, subsample=subsample, max_depth=depth, loss=loss_func, criterion=criteria, min_samples_leaf=min_samples_leaf, learning_rate=lr_rate, min_samples_split=2, verbose=1, random_state=1234, alpha=0.9, max_leaf_nodes=None, validation_fraction=0.1, n_iter_no_change=5)


    if tree_name is 'random_forest_clf':

    # criteria: ['gini','entropy']
        rf_clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criteria, max_depth=depth, max_features=max_features,  min_samples_leaf=min_samples_leaf, n_jobs=-1, min_samples_split=2,bootstrap=True, max_leaf_nodes=None, oob_score=False, verbose=1, random_state=1234, class_weight='balanced_subsample')

    if tree_name is 'random_forest_reg':

    # criteria: ['mae','mse']
        rf_reg = RandomForestRegressor(n_estimators=n_estimators, criterion=criteria, max_depth=depth, max_features=max_features, min_samples_leaf=min_samples_leaf, n_jobs=-1, min_samples_split=2,  bootstrap=True, max_leaf_nodes=None, oob_score=False,verbose=1, random_state=1234)

    return 




def tree(criteria, depth, n_estimators, lr_rate=0.1, max_features=0.7, min_samples_leaf=3, max_samples=0.75, subsample=0.7, loss_func='mse'):
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
    if tree_name = 'decision_tree_clf':
    dtc = DecisionTreeClassifier(criterion=criteria, max_depth=depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=2,random_state=1234, max_leaf_nodes=None, class_weight='balanced')

    if tree_name = 'decision_tree_reg':
    dtr = DecisionTreeRegressor(criterion=criteria, max_depth=depth, max_features=max_features,  min_samples_leaf=min_samples_leaf, min_samples_split=2, random_state=1234, max_leaf_nodes=None)


    if tree_name = 'ada_boost_clf':
    ada_noost = AdaBoostClassifier(base_estimator=None, n_estimators=n_estimators, learning_rate=lr_rate, random_state=1234)


    if tree_name = 'ada_boost_reg':
    # loss_func = {‘linear’, ‘square’, ‘exponential’}
    AdaBoostRegressor(base_estimator=None, n_estimators=n_estimators, loss=loss_func, learning_rate=lr_rate, random_state=1234)


    if tree_name = 'bagging_clf':
    # bootstrap_features: Whether features are drawn with replacement.
    bag_clf = BaggingClassifier(base_estimator=None, n_estimators=n_estimators,  max_samples=max_samples, max_features=max_features, bootstrap=True, bootstrap_features=False, oob_score=False, n_jobs=-1, random_state=1234, verbose=1)


    if tree_name = 'bagging_reg':
    bag_reg = BaggingRegressor(base_estimator=None, n_estimators=n_estimators,  max_samples=max_samples, max_features=max_features, bootstrap=True, bootstrap_features=False, oob_score=False, n_jobs=-1, random_state=1234, verbose=1)


    if tree_name = 'extra_tree_clf':
    """When looking for the best split to separate the samples 
    of a node into two groups, random splits are drawn for each 
    of the max_features randomly selected features and the best 
    split among those is chosen"""

    criteria: ['mae','mse']
    et_clf = ExtraTreesClassifier( criterion=criteria, max_depth=depth, max_features=max_features, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=2, n_jobs=-1, bootstrap=True, max_leaf_nodes=None, oob_score=False, verbose=1,random_state=1234, class_weight='balanced')

    if tree_name = 'extra_tree_reg':
    et_reg = ExtraTreesRegressor( criterion=criteria, max_depth=depth, max_features=max_features, n_estimators=n_estimators,min_samples_leaf=min_samples_leaf, min_samples_split=2, n_jobs=-1, bootstrap=False,  max_leaf_nodes=None, oob_score=False, verbose=1,random_state=1234)



    if tree_name = 'grad_boost_clf':

    # loss : {‘deviance’, ‘exponential’} (default=’deviance’)
    # criteria: Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error
    grad_boost_clf = GradientBoostingClassifier(max_features=max_features, n_estimators=n_estimators, subsample=subsample, max_depth=depth, loss=loss_func, criterion=criteria, min_samples_leaf=min_samples_leaf, learning_rate=lr_rate, min_samples_split=2, verbose=1, random_state=1234, max_leaf_nodes=None, validation_fraction=0.1, n_iter_no_change=5)


    if tree_name = 'grad_boost_reg':

    # loss_func: {‘ls’, ‘lad’, ‘huber’, ‘quantile’} (default='ls')
    # criteria: Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error
    # alpha: The alpha-quantile of the huber loss function and the quantile loss function. Only if loss='huber' or loss='quantile'
    grad_boost_reg = GradientBoostingRegressor( max_features=max_features, n_estimators=n_estimators, subsample=subsample, max_depth=depth, loss=loss_func, criterion=criteria, min_samples_leaf=min_samples_leaf, learning_rate=lr_rate, min_samples_split=2, verbose=1, random_state=1234, alpha=0.9, max_leaf_nodes=None, validation_fraction=0.1, n_iter_no_change=5)


    if tree_name = 'random_forest_clf':

    # criteria: ['gini','entropy']
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criteria, max_depth=depth, max_features=max_features,  min_samples_leaf=min_samples_leaf, n_jobs=-1, min_samples_split=2,bootstrap=True, max_leaf_nodes=None, oob_score=False, verbose=1, random_state=1234, class_weight='balanced_subsample')

    if tree_name = 'random_forest_reg':

    # criteria: ['mae','mse']
    rf_reg = RandomForestRegressor(n_estimators=n_estimators, criterion=criteria, max_depth=depth, max_features=max_features, min_samples_leaf=min_samples_leaf, n_jobs=-1, min_samples_split=2,  bootstrap=True, max_leaf_nodes=None, oob_score=False,verbose=1, random_state=1234)







"""
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










criteria = ['mae','mse']
etr = ExtraTreeRegressor(criterion=criteria, max_depth=depth, min_samples_split=2, min_samples_leaf=1, max_features=max_features, random_state=1234, max_leaf_nodes=None)






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










