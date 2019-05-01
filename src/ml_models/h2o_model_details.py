

##################################################### GLM ###################################################################
glm_model_inter = H2OGeneralizedLinearEstimator(
nfolds=3, seed=1234,
keep_cross_validation_predictions=True,
fold_assignment = 'stratified',
ignored_columns = drop_cols,
interactions = inter_cols,
family="binomial", 
lambda_search=True,
balance_classes=True,
remove_collinear_columns = True
)

glm_params = {
    'keep_cross_validation_predictions':True,
    'nfolds'              : 3,
    'ignored_columns'     : drop_cols,
    'seed'                : 1234,
    'distribution'        : 'AUTO', # Classification: binomial(binary), quasibinomial(binary), multinomial(Categorical)
                                   # numeric: poisson, laplace, tweedie, gaussian, huber, gamma, quantile
    'categorical_encoding': 'AUTO', #[AUTO, enum, enum_limited, one_hot_explicit, binary, eigen, label_encoder, sort_by_response (Reorders the levels by the mean response)]
    'histogram_type'      : 'AUTO', # [AUTO, UniformAdaptive, Random ==> (Extremely Randomized Trees), QuantilesGlobal, RoundRobin]
    'score_each_iteration': True, # scoring at each iteration
    'score_tree_interval' : 5, # score after each 5 tree built
    'fold_assignment'     : 'Stratified', # (used only is fold_column is not specified) [Random, Modulo, Stratified]
    'fold_column'         : None, # col name for cv fold
    'weights_column'      : col_name, # which should be present in the dataframe as an indiaction to weights of each row.
    'balance_classes'     : True, # only for classification (balance the classes by oversampling),
    
    'solver'              : 'AUTO', # (AUTO, IRLSM, L_BFGS, COORDINATE_DESCENT_NAIVE, COORDINATE_DESCENT, GRADIENT_DESCENT_LH, or GRADIENT_DESCENT_SQERR)
    # IRLSM is fast on problems with a small number of predictors and for lambda search with L1 penalty, while L_BFGS scales better for datasets with many columns. COORDINATE_DESCENT is IRLSM with the covariance updates version of cyclical coordinate descent in the innermost loop. COORDINATE_DESCENT_NAIVE is IRLSM with the naive updates version of cyclical coordinate descent in the innermost loop. GRADIENT_DESCENT_LH and GRADIENT_DESCENT_SQERR can only be used with the Ordinal family.
    'alpha'               : 0.2, # elastic net panlity (here, i used for less focus on l1)
    'lambda'              : 0.1, # regularization strength
    'lambda_search'       : True, # starting with lambda max (the smallest λ that drives all coefficients to zero). If you also specify a value for lambda_min_ratio, then this value is interpreted as lambda min.
    'lambda_min_ratio'    : 0.001,
    'early_stopping'      : True,
    'nlambdas'            : 100, # default, work only if lambda_search is enabled
    'standardize'         : True, #(default: mean=0, std=1)
	'family':'binomial', 
    # family: Specify the model type.
    #     If the family is gaussian, the response must be numeric (Real or Int). (default)
    #     If the family is binomial, the response must be categorical 2 levels/classes or binary (Enum or Int).
    #     If the family is multinomial, the response can be categorical with more than two levels/classes (Enum).
    #     If the family is ordinal, the response must be categorical with at least 3 levels.
    #     If the family is quasibinomial, the response must be numeric.
    #     If the family is poisson, the response must be numeric and non-negative (Int).
    #     If the family is negativebinomial, the response must be numeric and non-negative (Int).
    #     If the family is gamma, the response must be numeric and continuous and positive (Real or Int).
	# theta: Theta value (equal to 1/r) for use with the negative binomial family. This value must be > 0 and defaults to 1e-10.


	'seed'              : 1234,
	'interaction_pairs' : None, # a list of tuple as [("CRSDepTime", "UniqueCarrier"), ("CRSDepTime", "Origin")]
	'interactions'      : interaction_col_list, # a columns list ['a','b','c']
	'max_iterations'    : 1000,
	'non_negative'      : True, # will helpful in blending or for meta learners
	'intercept'         : True, # whether to include a bias term
	'compute_p_values'  : True, # Only applicable with no penalty (lambda = 0 and no beta constraints). Setting remove_collinear_columns is recommended. H2O will return an error if p-values are requested and there are collinear columns and remove_collinear_columns flag is not enabled. Note that this option is not available for family="multinomial" or family="ordinal".
	'link'              : 'logit', # (Identity, Family_Default, Logit, Log, Inverse, Tweedie, Ologit, Oprobit, and Ologlog)
    # If the family is Gaussian, then Identity, Log, and Inverse are supported.
    # If the family is Binomial, then Logit is supported.
    # If the family is Poisson, then Log and Identity are supported.
    # If the family is Gamma, then Inverse, Log, and Identity are supported.
    # If the family is Tweedie, then only Tweedie is supported.
    # If the family is Multinomial, then only Family_Default is supported. (This defaults to multinomial.)
    # If the family is Quasibinomial, then only Logit is supported.
    # If the family is Ordinal, then only Ologit, Oprobit, and Ologlog are supported. (Note that only Ologit is available for Ordinal regression.)
    # If the family is Negative Binomial, then only Log and Identity are supported.
    'prior'            : -1, # prior probability for p(y==1). Use this parameter for logistic regression if the data has been sampled and the mean of response does not reflect reality. This value defaults to -1 and must be a value in the range (0,1).
    # Note: This is a simple method affecting only the intercept. You may want to use weights and offset for a better fit.
	'remove_collinear_columns': True, # This can only be set if there is no regularization (lambda=0)
    'missing_values_handling' : 'MeanImputation', # [Skip, MeanImputation]

}
##################################################### GLM ###################################################################






################################################# GridSearch ###################################################################

"""
hyperparameters: a dict of parameters and a list f values for each parameter
search_criteria: a dict of values for following params. 
	(strategy, max_models, max_runtime_secs, stopping_metric, stopping_tolerance, stopping_rounds and seed)
strategy: cartesian, # [RandomDiscrete (random select), cartesian (all case)]
	cartesian: all possible case
	RandomDiscrete: Randomly choose some parameters, use max_models, seed, max_runtime_secs, stopping_rounds, stopping_metric, stopping_tolerance as 
	{'strategy': "RandomDiscrete", 'stopping_metric': "misclassification", 'stopping_tolerance': 0.0005, 'stopping_rounds': 5}
"""

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

gbm_params1 = { 'learn_rate': [i * 0.01 for i in range(1, 11)],
                'max_depth': list(range(2, 11)),
                'sample_rate': [0.8, 1.0],
                'col_sample_rate': [i * 0.1 for i in range(1, 11)]}

search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 36, 'seed': 1}

gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          grid_id='gbm_grid1',
                          hyper_params=gbm_params1,
                          search_criteria=None) # Either leave it or use as strategy
gbm_grid1.train(x=x, y=y,
                training_frame=train,
                validation_frame=valid,
                ntrees=100,
                seed=1)

# Get the grid results, sorted by validation AUC
gbm_gridperf1 = gbm_grid1.get_grid(sort_by='auc', decreasing=True)

best_gbm1 = gbm_gridperf1.models[0]
best_gbm_perf1 = best_gbm1.model_performance(test)
best_gbm_perf1.auc()

################################################# GridSearch ###################################################################




################################################# GBM ###################################################################

gbm_model_deep = H2OGradientBoostingEstimator(  
nfolds=3, seed=1234,
fold_assignment = 'stratified',
#                                                 score_validation_sampling='stratified',
ntrees = 10000,
learn_rate = 0.05,
max_depth = 4,
stopping_rounds = 5, 
stopping_tolerance = 1e-3,
stopping_metric = "AUC",
sample_rate = 0.7,
col_sample_rate = 0.7,
keep_cross_validation_predictions=True
)

gbm_params = {
	'nfolds'              :  3,
	'ignored_columns'     : drop_cols,
    'ntrees'              : 200,
	'max_depth'           : 10,# (default=20)
	'min_rows'            : None, # Specify the minimum number of observations for a leaf
	'nbins'               :  63, # Specify the number of bins for the histogram to build, then split at the best point.
	'nbins_cats'          : None # (Extensively tuning needed)
	'verbose'             : 25,
	'seed'                : 1234,
	'learn_rate'          : 0.01,
	'learn_rate_annealing': 0.99, # (danger as it reduce the lr_rate rapidly) (for lr:0.01, use lr:0.05, with anealing of 0.99, lead to better converger (fast))
	'distribution'        :'AUTO', # Classification: binomial(binary), quasibinomial(binary), multinomial(Categorical) and for numeric: poisson, laplace, tweedie, gaussian, huber, gamma, quantile
	'sample_rate'         : 0.7, # default 0.63 (samples without replacement)
    'sample_rate_per_class':0.7, # sample from the full dataset using a per-class-specific sampling rate rather than a global sample factor
    'col_sample_rate'     : 0.7, # sampling without replacement
    'col_sample_rate_per_tree':0.7, # sample without replacement.
	'histogram_type'      : 'AUTO', # [AUTO, UniformAdaptive, Random ==> (Extremely Randomized Trees), QuantilesGlobal, RoundRobin]
	'fold_column'         : None, # col name for cv fold
	'weights_column'      : col_name, # which should be present in the dataframe as an indiaction to weights of each row.
	'fold_assignment'     : 'Random', # (used only is fold_column is not specified) [Random, Modulo, Stratified]
	'balance_classes'     : True, # only for classification (balance the classes by oversampling),

	'min_split_improvement' : 1e-5, # need extensive tuning (the minimum relative improvement in squared error reduction in order for a split to happen. When properly tuned, this option can help reduce overfitting. Optimal values would be in the 1e-10…1e-3 range.)
	'categorical_encoding'  : 'AUTO', #[AUTO, enum, enum_limited, one_hot_explicit, binary, eigen, label_encoder, sort_by_response (Reorders the levels by the mean response)]
    'keep_cross_validation_predictions':True,
	'score_each_iteration'  : True, # scoring at each iteration
	'score_tree_interval'   : 5, # score after each 5 tree built
	'stopping_rounds'       : 25, # wait for n(25) itrs for early stopping
	'stopping_metric'       : 'auc', # [deviance, logloss, mse, rmse, mae, rmsle, auc, misclassification, mean_per_class_error]
	'stopping_tolerance'    : 0.001, # tolerance factor for wait till stopping
	'max_after_balance_size': 1,# (0-inf) for oversampling choose > 1, else < 1.
	'class_sampling_factors': 1, # ration of over/under-sampling rate. By default, these ratios are automatically computed during training to obtain the class balance. Note that this requires balance_classes=true.
	'quantile_alpha'        : 0.01, # when distribution is quantile. (Specify the quantile to be used for Quantile Regression.)
	'huber_alpha'           : 0.001, # Huber/M-regression (the threshold between quadratic and linear loss)
	'max_abs_leafnode_pred' : None, # (only for clf), it reduce overfitting by limiting the maximum absolute value of a leaf node prediction
	'pred_noise_bandwidth'  : 0 # The bandwidth (sigma) of Gaussian multiplicative noise ~N(1,sigma) for tree node predictions. If this parameter is specified with a value greater than 0, then every leaf node prediction is randomly scaled by a number drawn from a Normal distribution centered around 1 with a bandwidth given by this parameter
	'nbins_top_level'       : None # Specify the minimum number of bins at the root level to use to build the histogram. This number will then be decreased by a factor of two per level.
}
"""
Leaf Node Assignment:

 # Use h2o.predict_leaf_node_assignment(model, frame) to get an H2OFrame with the leaf node assignments. Those leaf nodes represent decision rules that can be fed to other models (i.e., GLM with lambda search and strong rules) to obtain a limited set of the most important rules.
"""
################################################# GBM ###################################################################





################################################# DRF ###################################################################

drf_model_deep = H2ORandomForestEstimator( nfolds=3, seed=1234,
                                           keep_cross_validation_predictions=True,
                                           fold_assignment = 'stratified',
                                           histogram_type = 'QuantilesGlobal',
                                           categorical_encoding = 'eigen',
                                           stopping_metric = 'auc',
                                           ntrees = 100,
                                           balance_classes = True
                                           )


drf_params = {
	'nfolds': 3,
	'keep_cross_validation_predictions':True,
	'score_each_iteration' : True, # scoring at each iteration
	'score_tree_interval'  : 5, # score after each 5 tree built
	'fold_assignment'      : 'Random', # (used only is fold_column is not specified) [Random, Modulo, Stratified]
	'fold_column'          : None, # col name for cv fold
	'ignored_columns'      : drop_cols,
	'balance_classes'      : True, # only for classification (balance the classes by oversampling),
	'max_after_balance_size':1,# (0-inf) for oversampling choose > 1, else < 1.
	'ntrees'               : 200,
	'max_depth'            : 10,# (default=20)
	'min_rows'             : None, # Specify the minimum number of observations for a leaf
	'nbins'                : 63, # Specify the number of bins for the histogram to build, then split at the best point.
	'nbins_top_level'      : None # Specify the minimum number of bins at the root level to use to build the histogram. This number will then be decreased by a factor of two per level.
	# 'nbins_cats': # (Extensively tuning needed)
	'stopping_rounds'      : 25, # wait for n(25) itrs for early stopping
	'stopping_metric'      : 'auc', # [deviance, logloss, mse, rmse, mae, rmsle, auc, misclassification, mean_per_class_error]
	'stopping_tolerance'   : 0.001, # tolerance factor for wait till stopping
	'seed'                 : 1234,
	'categorical_encoding' : 'AUTO', #[AUTO, enum, enum_limited, one_hot_explicit, binary, eigen, label_encoder, sort_by_response (Reorders the levels by the mean response)]
	'verbose'              : 25,
    'histogram_type'       : 'AUTO', # [AUTO, UniformAdaptive, Random ==> (Extremely Randomized Trees), QuantilesGlobal, RoundRobin]
	'col_sample_rate_per_tree':0.7, # sample without replacement.
    'min_split_improvement': 1e-5, # need extensive tuning (the minimum relative improvement in squared error reduction in order for a split to happen. When properly tuned, this option can help reduce overfitting. Optimal values would be in the 1e-10…1e-3 range.)
    'sample_rate'          : 0.7, # default 0.63 (samples without replacement)
    'sample_rate_per_class': 0.7, # sample from the full dataset using a per-class-specific sampling rate rather than a global sample factor
    'binomial_double_trees': True, # (Binary classification only) Build twice as many trees (one per class). Enabling this option can lead to higher accuracy, while disabling can result in faster model building.
    'mtries'               : -1, # Specify the columns to randomly select at each level. If the default value of -1 is used, the number of variables is the square root of the number of columns for classification and p/3 for regression (where p is the number of predictors). The range is -1 to >=1.
    'class_sampling_factors':1, # ration of over/under-sampling rate. By default, these ratios are automatically computed during training to obtain the class balance. Note that this requires balance_classes=true.
    'weights_column':col_name, # which should be present in the dataframe as an indiaction to weights of each row.
}

"""
    Does the algo stop splitting when all the possible splits lead to worse error measures?
    	It does if you use min_split_improvement (min_split_improvement turned ON by default (0.00001).) When properly tuned, this option can help reduce overfitting.
    

    How does DRF decide which feature to split on?
  		It splits on the column and level that results in the greatest reduction in residual sum of the squares (RSS) in the subtree at that point. It considers all fields available from the algorithm. Note that any use of column sampling and row sampling will cause each decision to not consider all data points, and that this is on purpose to generate more robust trees. To find the best level, the histogram binning process is used to quickly compute the potential MSE of each possible split. The number of bins is controlled via nbins_cats for categoricals, the pair of nbins (the number of bins for the histogram to build, then split at the best point), and nbins_top_level (the minimum number of bins at the root level to use to build the histogram). This number will then be decreased by a factor of two per level.

    For nbins_top_level, higher = more precise, but potentially more prone to overfitting. Higher also takes more memory and possibly longer to run.

    What is the difference between nbins and nbins_top_level?
    	nbins and nbins_top_level are both for numerics (real and integer). nbins_top_level is the number of bins DRF uses at the top of each tree. It then divides by 2 at each ensuing level to find a new number. nbins controls when DRF stops dividing by 2.
    
    binomial_double_trees: (Binary classification only) Build twice as many trees (one per class). Enabling this option can lead to higher accuracy, while disabling can result in faster model building. This option is disabled by default.
"""
################################################# DRF ###################################################################




################################################# Kmean ###################################################################

kmeans_model = H2OKMeansEstimator( k=2, 
#                                    fold_assignment = 'stratified',
                                   keep_cross_validation_predictions=True, 
                                   nfolds = 3,
                                   ignored_columns = drop_cols,
                                   seed=1234,
                                   categorical_encoding = 'one_hot_explicit',
                                   estimate_k = True,
                                   max_iterations = 1000
                                   
                                  )
kmean_params = {
	'keep_cross_validation_predictions':True,
	'ignored_columns'     : drop_cols, # a list of columns to be dropped
	'score_each_iteration': True, # used when early_stopping is used
	'k'                   : 3, # no of clusters
	# estimate_k ==> whether to estimate the number of clusters (<=k) iteratively (independent of the seed) and deterministically
	'estimate_k'          : True, 
	'max_iterations'      : 1000, # (0 to 1e6)
	'standardize'         : True,
	'seed'                : 1234
	'init'                : 'Furthest', #[Random, Furthest, PlusPlus]
	'categorical_encoding': 'AUTO',# [AUTO, enum(1 col), one_hot_explicit(N+1), binary, eigen(k cols)]
}
################################################# Kmean ###################################################################




################################################# low rank ###################################################################

from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
low_rank = H2OGeneralizedLowRankEstimator(k=5,#gamma_x=0.1, gamma_y=0.1,
                                          regularization_x='l2',
                                          regularization_y='l1',
                                          seed=1234
                                   )

low_rank_params = {
	'ignored_columns'      : drop_cols,
	'score_each_iteration' : True,
	'transform'            : 'Standardize', # [None, Standardize, Normalize, Demean, Descale]
	'k'                    : 5, # rank of marrix
	'loss'                 : 'Quadratic', # [Quadratic, Absolute, Huber, Poisson, Hinge]
	'max_iterations'       : 1000, # [1, 1e6]
	'gamma_x'              : 0.001, # reg weights on X matrix
	'gamma_y'              : 0.001, # reg weights on Y matrix
	'multi_loss'           : 'Categorical', # When it is defined, then it will treat cat_var differently. [Ordinal, Categorical]
	'init'                 : 'SVD', # [Random, SVD, PlusPlus]
	'init_step_size'       : 1, # initial step size (not clear, how it works)
	'min_step_size'        : 1e-6, # min step size (not clear, how to choose)
	'seed'                 : 1234,
	'svd_method'           : 'GramSVD', # [GramSVD, Power, Randomized(not stable in current version)]
	'impute_original'      : False, # This is whole idea of using this algorithm, to fill those Nan values
	'recover_svd'          : False, # whether to recover singular values and eigenvectors of XY.
	'regularization_x'     : 'l2', # [None, Quadratic, L2, L1, NonNegative, OneSparse, UnitOneSparse, Simplex]
	'regularization_y'     : 'l2', # [None, Quadratic, L2, L1, NonNegative, OneSparse, UnitOneSparse, Simplex]
}

################################################# low rank ###################################################################









################################################# PCA ###################################################################

from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator

pca_model = H2OPrincipalComponentAnalysisEstimator(k=3,
                                                   seed=1234,
                                                   max_iterations=5000,
                                                   transform='standardize',
                                                   ignored_columns = drop_cols
                                                   )

pca_params = {
	'k':4, # rank of matrix approx [range 1-9]
	'ignored_columns':drop_cols,
	'transform':'Standardize', # transformation of training data, [Standardize, Normalize, Demean, or Descale]
	'pca_method':'GramSVD', # [GramSVD, Power, Randomized, GLRM]
    # GramSVD: Uses a distributed computation of the Gram matrix, followed by a local SVD using the JAMA package
    # Power: Computes the SVD using the power iteration method (experimental)
    # Randomized: Uses randomized subspace iteration method
    # GLRM: Fits a generalized low-rank model with L2 loss function and no regularization and solves for the SVD using local matrix algebra (experimental)
    'max_iterations':2000, #[1, 1e6]
    'seed':1234,
    'compute_metrics':True # Enable metrics computations on the training data.
    'score_each_iteration':True,
    'impute_missing':True, # fill missing values with mean value
    'pca_impl':'mtj_svd_densematrix'
    # mtj_evd_densematrix: Eigenvalue decompositions for dense matrix using Matrix Toolkit Java (MTJ)
    # mtj_evd_symmmatrix: Eigenvalue decompositions for symmetric matrix using Matrix Toolkit Java (MTJ) (default)
    # mtj_svd_densematrix: Singular-value decompositions for dense matrix using Matrix Toolkit Java (MTJ)
    # jama: Eigenvalue decompositions for dense matrix using Java Matrix (JAMA)

}

    # When running PCA, is it better to create a cluster that uses many smaller nodes or fewer larger nodes?

    # For PCA, this is dependent on the specified pca_method parameter:

    #     For GramSVD, use fewer larger nodes for better performance. Forming the Gram matrix requires few intensive calculations and the main bottleneck is the JAMA library’s SVD function, which is not parallelized and runs on a single machine. We do not recommend selecting GramSVD for datasets with many columns and/or categorical levels in one or more columns.
    #     For Randomized, use many smaller nodes for better performance, since H2O calls a few different distributed tasks in a loop, where each task does fairly simple matrix algebra computations.
    #     For GLRM, the number of nodes depends on whether the dataset contains many categorical columns with many levels. If this is the case, we recommend using fewer larger nodes, since computing the loss function for categoricals is an intensive task. If the majority of the data is numeric and the categorical columns have only a small number of levels (~10-20), we recommend using many small nodes in the cluster.
    #     For Power, we recommend using fewer larger nodes because the intensive calculations are single-threaded. However, this method is only recommended for obtaining principal component values (such as k << ncol(train)) because the other methods are far more efficient.

################################################# PCA ###################################################################
