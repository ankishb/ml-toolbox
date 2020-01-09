


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








class GradientBoostingClassifier(ClassifierMixin, BaseGradientBoosting):
    """Gradient Boosting for classification.
    GB builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage ``n_classes_``
    regression trees are fit on the negative gradient of the
    binomial or multinomial deviance loss function. Binary classification
    is a special case where only a single regression tree is induced.
    Read more in the :ref:`User Guide <gradient_boosting>`.
    Parameters
    ----------
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.
    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.
        .. versionadded:: 0.18
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
        .. versionadded:: 0.19
    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.
    init : estimator or 'zero', optional (default=None)
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :meth:`fit` and :meth:`predict_proba`. If
        'zero', the initial raw predictions are set to zero. By default, a
        ``DummyEstimator`` predicting the classes priors is used.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.
    presort : deprecated, default='deprecated'
        This parameter is deprecated and will be removed in v0.24.
        .. deprecated :: 0.22
    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.
        .. versionadded:: 0.20
    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.
        .. versionadded:: 0.20
    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        .. versionadded:: 0.20
    ccp_alpha : non-negative float, optional (default=0.0)
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.
        .. versionadded:: 0.22
    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.
        .. versionadded:: 0.20
    feature_importances_ : array, shape (n_features,)
        The feature importances (the higher, the more important the feature).
    oob_improvement_ : array, shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``
    train_score_ : array, shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.
    loss_ : LossFunction
        The concrete ``LossFunction`` object.
    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.
    estimators_ : ndarray of DecisionTreeRegressor,\
shape (n_estimators, ``loss_.K``)
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.
    classes_ : array of shape (n_classes,)
        The classes labels.
    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.
    See also
    --------
    sklearn.ensemble.HistGradientBoostingClassifier,
    sklearn.tree.DecisionTreeClassifier, RandomForestClassifier
    AdaBoostClassifier
    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
    J. Friedman, Stochastic Gradient Boosting, 1999
    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    """

    _SUPPORTED_LOSS = ('deviance', 'exponential')

    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='deprecated', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start, presort=presort,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)


class GradientBoostingRegressor(RegressorMixin, BaseGradientBoosting):
    """Gradient Boosting for regression.
    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the
    given loss function.
    Read more in the :ref:`User Guide <gradient_boosting>`.
    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'quantile'}, optional (default='ls')
        loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).
    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.
        .. versionadded:: 0.18
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
        .. versionadded:: 0.19
    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.
        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.
    init : estimator or 'zero', optional (default=None)
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :term:`fit` and :term:`predict`. If 'zero', the
        initial raw predictions are set to zero. By default a
        ``DummyEstimator`` is used, predicting either the average target value
        (for loss='ls'), or a quantile for the other losses.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    alpha : float (default=0.9)
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.
    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.
    presort : deprecated, default='deprecated'
        This parameter is deprecated and will be removed in v0.24.
        .. deprecated :: 0.22
    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.
        .. versionadded:: 0.20
    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.
        .. versionadded:: 0.20
    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        .. versionadded:: 0.20
    ccp_alpha : non-negative float, optional (default=0.0)
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.
        .. versionadded:: 0.22
    Attributes
    ----------
    feature_importances_ : array, shape (n_features,)
        The feature importances (the higher, the more important the feature).
    oob_improvement_ : array, shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``
    train_score_ : array, shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.
    loss_ : LossFunction
        The concrete ``LossFunction`` object.
    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.
    estimators_ : array of DecisionTreeRegressor, shape (n_estimators, 1)
        The collection of fitted sub-estimators.
    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.
    See also
    --------
    sklearn.ensemble.HistGradientBoostingRegressor,
    sklearn.tree.DecisionTreeRegressor, RandomForestRegressor
    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
    J. Friedman, Stochastic Gradient Boosting, 1999
    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    """

    _SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile')

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='deprecated',
                 validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, alpha=alpha, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            presort=presort, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)
