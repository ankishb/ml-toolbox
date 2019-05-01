






from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
# from h2o.estimators.kmeans import H2OKMeansEstimator
# from h2o.estimators.xgboost import H2OXGBoostEstimator



dl_model = H2ODeepLearningEstimator(epochs=10, nfolds=3, seed=1234, 
                                    score_validation_sampling='stratified',
                                    keep_cross_validation_predictions=True,
                                    balance_classes = True,
                                    l2 = 0.0001,
                                    loss = 'cross_entropy',
#                                     stopping_rounds=2,
                                    mini_batch_size = 200,
                                    rate_decay=0.1,
                                    stopping_rounds=2
                                   )

dl_model_deep = H2ODeepLearningEstimator( nfolds=3, seed=1234,
                                          score_validation_sampling='stratified',
                                          keep_cross_validation_predictions=True,
                                          hidden=[100,100],
                                          epochs=50,
                                          balance_classes = True,
                                          l2 = 0.0001,
                                          loss = 'cross_entropy',
                                          mini_batch_size = 200,
                                          rate_decay=0.1,
                                          stopping_rounds=2,
#                                           score_validation_samples=10000,
                                          stopping_metric="auc",
                                          stopping_tolerance=0.01
                                        )


gbm_model = H2OGradientBoostingEstimator(nfolds=3, seed=1234,
                                         balance_classes = True,
                                         col_sample_rate = 0.7,
                                         learn_rate=0.1,
                                         nbins = 128,
                                         fold_assignment='stratified',
                                         stopping_rounds=25,
                                         keep_cross_validation_predictions=True
                                        )

gbm_model_deep = H2OGradientBoostingEstimator(  nfolds=3, seed=1234,
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

drf_model = H2ORandomForestEstimator(nfolds=3, seed=1234,
                                     ntrees=200,
#                                      score_validation_sampling='stratified',
                                     keep_cross_validation_predictions=True,
                                     fold_assignment = 'stratified',
                                     categorical_encoding='enum',
                                     histogram_type='round_robin',
                                     max_depth = 5
                                    )

glm_model = H2OGeneralizedLinearEstimator(nfolds=3, seed=1234,
                                          keep_cross_validation_predictions=True,
                                          fold_assignment = 'stratified',
                                          family="binomial")

glm_model_bal = H2OGeneralizedLinearEstimator(nfolds=3, seed=1234,
                                              keep_cross_validation_predictions=True,
                                              fold_assignment = 'stratified',
                                              family="binomial", 
                                              lambda_search=True,
                                              balance_classes=True)

glm_model_inter = H2OGeneralizedLinearEstimator(nfolds=3, seed=1234,
                                                keep_cross_validation_predictions=True,
                                                fold_assignment = 'stratified',
                                                ignored_columns = drop_cols,
#                                                 interactions = inter_cols,
                                                family="binomial", 
                                                lambda_search=True,
                                                balance_classes=True,
                                                remove_collinear_columns = True)

########################################################################################
########################################################################################
########################################################################################












########################################################################################
########################################################################################
########################################################################################

# Advance gfeature engineering: h2o.predict_leaf_node_assignment( model, frame )




    offset_column
    weights_column
    nbins_top_level
    nbins_cats
    nbins
    ntrees
    max_depth
    min_rows
    learn_rate
    r2_stopping
    stopping_rounds
    stopping_metric
    stopping_tolerance
    max_runtime_secs
    learn_rate_annealing
    quantile_alpha
    tweedie_power
    huber_alpha







AutoML Parameters:
GLM Hyperparameters

    alpha
    missing_values_handling

XGBoost Hyperparameters

    ntrees
    max_depth
    min_rows
    min_sum_hessian_in_leaf
    sample_rate
    col_sample_rate
    col_sample_rate_per_tree
    booster
    reg_lambda
    reg_alpha

GBM Hyperparameters

    histogram_type
    ntrees
    max_depth
    min_rows
    learn_rate
    sample_rate
    col_sample_rate
    col_sample_rate_per_tree
    min_split_improvement

Deep Learning Hyperparameters

    epochs
    adaptivate_rate
    activation
    rho
    epsilon
    input_dropout_ratio
    hidden
    hidden_dropout_ratios

########################################################################################
########################################################################################
########################################################################################







########################################################################################
########################################################################################
########################################################################################

glm_model_inter = H2OGeneralizedLinearEstimator(
nfolds=3, seed=1234,
keep_cross_validation_predictions=True,
fold_assignment = 'stratified',
ignored_columns = drop_cols,
#                                                 interactions = inter_cols,
family="binomial", 
lambda_search=True,
balance_classes=True,
remove_collinear_columns = True)

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







    


########################################################################################
########################################################################################
########################################################################################





########################################################################################
########################################################################################
########################################################################################


dl_model_deep = H2ODeepLearningEstimator( nfolds=3, seed=1234,
score_validation_sampling='stratified',
keep_cross_validation_predictions=True,
hidden=[100,100],
epochs=50,
balance_classes = True,
l2 = 0.0001,
loss = 'cross_entropy',
mini_batch_size = 200,
rate_decay=0.1,
stopping_rounds=2,
#                                           score_validation_samples=10000,
stopping_metric="auc",
stopping_tolerance=0.01
)


deep_ml_params = {
	'nfolds': 3,
	'keep_cross_validation_predictions':True,
	'score_each_iteration':True, # scoring at each iteration
	'fold_assignment':'Random', # (used only is fold_column is not specified) [Random, Modulo, Stratified]
	'fold_column':None, # col name for cv fold
	'ignored_columns':drop_cols,
	'weights_column':col_name, # which should be present in the dataframe as an indiaction to weights of each row.
	'balance_classes':True, # only for classification (balance the classes by oversampling),
	'class_sampling_factors':1, # ration of over/under-sampling rate. By default, these ratios are automatically computed during training to obtain the class balance. Note that this requires balance_classes=true.
	'pretrained_autoencoder':'auto_enc_model', # a pretrained autoencoder model to initialize this model with.
	'standardize':True, # mean=0, std=1
	'activation':'Rectifier', # [Tanh, Tanh with dropout, Rectifier, Rectifier with dropout]
	'hidden':(100,100), # hidden layers size
	'epochs':100,
	'distribution':'AUTO', # [clf: bernoulli, multinomial], [reg: gaussian, poisson, gamma, laplace, quantile, huber, or tweedie.]
	'mini_batch_size':250,
	'categorical_encoding':'AUTO', #[AUTO, enum, enum_limited, one_hot_explicit, binary, eigen, label_encoder, sort_by_response (Reorders the levels by the mean response)]
	'verbose':25,
	'max_after_balance_size':1,# (0-inf) for oversampling choose > 1, else < 1.
	'overwrite_with_best_model':True, # overwrite the final model with the best model found during training (enabled by default).
	'seed':1234,
    'missing_values_handling':'MeanImputation', # [Skip, MeanImputation]
    'stopping_rounds':25, # wait for n(25) itrs for early stopping
	'stopping_metric':'auc', # [deviance, logloss, mse, rmse, mae, rmsle, auc, misclassification, mean_per_class_error]
	'stopping_tolerance':0.001, # tolerance factor for wait till stopping
	'l1':0.001, # l1 regularization
	'l2':0.01, #l2 regularization
	'max_w2':0.001, #the constraint for the squared sum of the incoming weights per unit (e.g., for Rectifier)
	'initial_weight_distribution':'Uniform', # [Uniform Adaptive, Uniform, or Normal]
	'initial_weight_scale': 0.0001, # (only if initial_weight_distribution is Uniform or Normal) Specify the scale of the distribution function. 
	'loss': 'AUTO', # [clf: CrossEntropy, Quadratic, Huber, or Absolute] [Reg: Quadratic, Huber, or Absolute]
	'huber_alpha':0.5, # (the threshold between quadratic and linear loss) [range: 0-1]
	'input_dropout_ratio':0.1, # [dropour on input layer]
	'hidden_dropout_ratios':(0.1, 0.1), # (only work, if activation is TanhWithDropout, RectifierWithDropout) (one value for each layer)
    
	# 'score_interval':0.1, # shortest time interval (in seconds) to wait between model scoring.
    'score_validation_samples':'stratified', # method used to sample validation dataset for scoring [uniform, stratified]

    'offset_column':None, # columns which represnt the bias value 



	'ntrees':200,
	'max_depth':10,# (default=20)
	'min_rows':None, # Specify the minimum number of observations for a leaf
	'nbins': 63, # Specify the number of bins for the histogram to build, then split at the best point.
	'nbins_top_level': # Specify the minimum number of bins at the root level to use to build the histogram. This number will then be decreased by a factor of two per level.
	# 'nbins_cats': # (Extensively tuning needed)
	
	
    'histogram_type':'AUTO', # [AUTO, UniformAdaptive, Random ==> (Extremely Randomized Trees), QuantilesGlobal, RoundRobin]
	'col_sample_rate_per_tree':0.7, # sample without replacement.
    'min_split_improvement':1e-5, # need extensive tuning (the minimum relative improvement in squared error reduction in order for a split to happen. When properly tuned, this option can help reduce overfitting. Optimal values would be in the 1e-10…1e-3 range.)
    'sample_rate':0.7, # default 0.63 (samples without replacement)
    'sample_rate_per_class':0.7, # sample from the full dataset using a per-class-specific sampling rate rather than a global sample factor
    'binomial_double_trees':True, # (Binary classification only) Build twice as many trees (one per class). Enabling this option can lead to higher accuracy, while disabling can result in faster model building.
    'mtries':-1, # Specify the columns to randomly select at each level. If the default value of -1 is used, the number of variables is the square root of the number of columns for classification and p/3 for regression (where p is the number of predictors). The range is -1 to >=1.
    
}


    




    use_all_factor_levels: Specify whether to use all factor levels in the possible set of predictors; if you enable this option, sufficient regularization is required. By default, the first factor level is skipped. For Deep Learning models, this option is useful for determining variable importances and is automatically enabled if the autoencoder is selected.

    train_samples_per_iteration: Specify the number of global training samples per MapReduce iteration. To specify one epoch, enter 0. To specify all available data (e.g., replicated training data), enter -1. To use the automatic values, enter -2.

    target_ratio_comm_to_comp: Specify the target ratio of communication overhead to computation. This option is only enabled for multi-node operation and if train_samples_per_iteration equals -2 (auto-tuning).


    adaptive_rate: Specify whether to enable the adaptive learning rate (ADADELTA). This option is enabled by default.

    rho: (Applicable only if adaptive_rate is enabled) Specify the adaptive learning rate time decay factor.

    epsilon:(Applicable only if adaptive_rate is enabled) Specify the adaptive learning rate time smoothing factor to avoid dividing by zero.

    rate: (Applicable only if adaptive_rate is disabled) Specify the learning rate. Higher values result in a less stable model, while lower values lead to slower convergence.

    rate_annealing: (Applicable only if adaptive_rate is disabled) Specify the rate annealing value. The rate annealing is calculated as rate(1 + rate_annealing * samples).

    rate_decay: (Applicable only if adaptive_rate is disabled) Specify the rate decay factor between layers. The rate decay is calculated as (N-th layer: rate * alpha^(N-1)).

    momentum_start: (Applicable only if adaptive_rate is disabled) Specify the initial momentum at the beginning of training; we suggest 0.5.

    momentum_ramp: (Applicable only if adaptive_rate is disabled) Specify the number of training samples for which the momentum increases.

    momentum_stable: (Applicable only if adaptive_rate is disabled) Specify the final momentum after the ramp is over; we suggest 0.99.

    nesterov_accelerated_gradient: (Applicable only if adaptive_rate is disabled) Enables the Nesterov Accelerated Gradient.



    classification_stop: This option specifies the stopping criteria in terms of classification error (1-accuracy) on the training data scoring dataset. When the error is at or below this threshold, training stops. To disable this option, enter -1.

    regression_stop: (Regression models only) Specify the stopping criterion for regression error (MSE) on the training data. When the error is at or below this threshold, training stops. To disable this option, enter -1.



    diagnostics: Specify whether to compute the variable importances for input features (using the Gedeon method). For large networks, enabling this option can reduce speed. This option is enabled by default.

    fast_mode: Specify whether to enable fast mode, a minor approximation in back-propagation. This option is enabled by default.

    force_load_balance: Specify whether to force extra load balancing to increase training speed for small datasets and use all cores. This option is enabled by default.

    variable_importances: Specify whether to compute variable importance. This option is not enabled by default.

    replicate_training_data: Specify whether to replicate the entire training dataset onto every node for faster training on small datasets.

    single_node_mode: Specify whether to run on a single node for fine-tuning of model parameters.

    shuffle_training_data: Specify whether to shuffle the training data. This option is recommended if the training data is replicated and the value of train_samples_per_iteration is close to the number of nodes times the number of rows. This option is not enabled by default.


    quiet_mode: Specify whether to display less output in the standard output. This option is not enabled by default.


    sparse: Specify whether to enable sparse data handling, which is more efficient for data with many zero values.

    col_major: Specify whether to use a column major weight matrix for the input layer. This option can speed up forward propagation but may reduce the speed of backpropagation. This option is not enabled by default.

    average_activation: Specify the average activation for the sparse autoencoder. If Rectifier is used, the average_activation value must be positive.

    sparsity_beta: (Applicable only if autoencoder is enabled) Specify the sparsity-based regularization optimization. For more information, refer to the following link.

    max_categorical_features: Specify the maximum number of categorical features enforced via hashing. The value must be at least one.

    reproducible: Specify whether to force reproducibility on small data. If this option is enabled, the model takes more time to generate because it uses only one thread.

    export_weights_and_biases: Specify whether to export the neural network weights and biases as H2O frames.



    elastic_averaging: Specify whether to enable elastic averaging between computing nodes, which can improve distributed model convergence.
    elastic_averaging_moving_rate: Specify the moving rate for elastic averaging. This option is only available if elastic_averaging=True.
    elastic_averaging_regularization: Specify the elastic averaging regularization strength. This option is only available if elastic_averaging=True.
    export_checkpoints_dir: Optionally specify a path to a directory where every generated model will be stored when checkpointing models.


########################################################################################
########################################################################################
########################################################################################










########################################################################################
########################################################################################
########################################################################################

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


########################################################################################
########################################################################################
########################################################################################










########################################################################################
########################################################################################
########################################################################################


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

Leaf Node Assignment:

 # Use h2o.predict_leaf_node_assignment(model, frame) to get an H2OFrame with the leaf node assignments. Those leaf nodes represent decision rules that can be fed to other models (i.e., GLM with lambda search and strong rules) to obtain a limited set of the most important rules.

########################################################################################
########################################################################################
########################################################################################












########################################################################################
########################################################################################
########################################################################################


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


    Does the algo stop splitting when all the possible splits lead to worse error measures?
    	It does if you use min_split_improvement (min_split_improvement turned ON by default (0.00001).) When properly tuned, this option can help reduce overfitting.
    

    How does DRF decide which feature to split on?
  		It splits on the column and level that results in the greatest reduction in residual sum of the squares (RSS) in the subtree at that point. It considers all fields available from the algorithm. Note that any use of column sampling and row sampling will cause each decision to not consider all data points, and that this is on purpose to generate more robust trees. To find the best level, the histogram binning process is used to quickly compute the potential MSE of each possible split. The number of bins is controlled via nbins_cats for categoricals, the pair of nbins (the number of bins for the histogram to build, then split at the best point), and nbins_top_level (the minimum number of bins at the root level to use to build the histogram). This number will then be decreased by a factor of two per level.

    For nbins_top_level, higher = more precise, but potentially more prone to overfitting. Higher also takes more memory and possibly longer to run.

    What is the difference between nbins and nbins_top_level?
    	nbins and nbins_top_level are both for numerics (real and integer). nbins_top_level is the number of bins DRF uses at the top of each tree. It then divides by 2 at each ensuing level to find a new number. nbins controls when DRF stops dividing by 2.
    
    binomial_double_trees: (Binary classification only) Build twice as many trees (one per class). Enabling this option can lead to higher accuracy, while disabling can result in faster model building. This option is disabled by default.

########################################################################################
########################################################################################
########################################################################################
    categorical_encoding: Specify one of the following encoding schemes for handling categorical features:
        auto or AUTO: Allow the algorithm to decide (default). In K-Means, the algorithm will automatically perform enum encoding.
        enum or Enum: 1 column per categorical feature
        one_hot_explicit: N+1 new columns for categorical features with N levels
        binary or Binary: No more than 32 columns per categorical feature
        `eigen or Eigen: k columns per categorical feature, keeping projections of one-hot-encoded matrix onto k-dim eigen space only
        label_encoder or LabelEncoder: Convert every enum into the integer of its index (for example, level 0 -> 0, level 1 -> 1, etc.)

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


########################################################################################
########################################################################################
########################################################################################













########################################################################################
########################################################################################
########################################################################################

dl_model_deep = H2ODeepLearningEstimator( nfolds=3, seed=1234,
                                          score_validation_sampling='stratified',
                                          keep_cross_validation_predictions=True,
                                          hidden=[100,10],
                                          activation='rectifier_with_dropout',
                                          fold_assignment='Stratified',
                                          epochs=500,
#                                           categorical_encoding='eigen',
                                          balance_classes = True,
                                          l2 = 0.0001,
                                          loss = 'cross_entropy',
                                          mini_batch_size = 500,
                                          rate_decay=0.1,
                                          stopping_rounds=3,
#                                           score_validation_samples=10000,
                                          stopping_metric="auc",
                                          stopping_tolerance=0.01
                                        )
# # categorical_encoding
# # "auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", 
# # "label_encoder", 
# # "sort_by_response", "enum_limited" (default: "auto").

gbm_model_one_hot = H2OGradientBoostingEstimator(nfolds=3, seed=1234,
                                         balance_classes = True,
                                         col_sample_rate = 0.7,
                                         learn_rate=0.1,
                                         ntrees = 10000,
#                                          nbins = 128,one_hot_internal
# sort_by_response
                                         categorical_encoding='one_hot_explicit',
                                         fold_assignment='stratified',
                                         stopping_rounds=25,
#                                          categorical_encoding='enum',
                                         keep_cross_validation_predictions=True
                                        )
drf_model = H2ORandomForestEstimator(nfolds=3, seed=1234,
                                     ntrees=200,
                                     balance_classes=True,
                                     nbins_cats=512,
#                                      score_validation_sampling='stratified',
                                     keep_cross_validation_predictions=True,
                                     fold_assignment = 'stratified',
#                                      categorical_encoding='enum',
#                                      histogram_type='round_robin',
                                     max_depth = 6
                                    )



xgboost_model = H2OXGBoostEstimator(ntrees = 10000,
                                    max_bins=128,
                                    col_sample_rate=0.7,
                                    fold_assignment = 'stratified',
                                    eta = 0.1,
                                    keep_cross_validation_predictions=True, 
                                    nfolds = 3,
                                    ignored_columns = drop_cols,
                                    seed=1234,
                                    categorical_encoding = 'one_hot_explicit',
                                    grow_policy = 'depthwise', #lossguide(lgbm)
                                    max_depth = 5,
                                    min_child_weight = 3,
                                    quiet_mode = False,
                                    reg_lambda = 1.5,
                                    sample_rate=0.7,
                                    stopping_metric='auc',
                                    stopping_rounds=25,
                                    subsample=.7
                                  )


########################################################################################
########################################################################################
########################################################################################

categorical_encoding

    Available in: GBM, DRF, Deep Learning, K-Means, Aggregator, XGBoost, Isolation Forest
    Hyperparameter: yes

Description

This option specifies the encoding scheme to use for handling categorical features. Available schemes include the following:

GBM/DRF/Isolation Forest

        auto or AUTO: Allow the algorithm to decide (default). For GBM, DRF, and Isolation Forest, the algorithm will perform Enum encoding when auto option is specified.
        enum or Enum: Leave the dataset as is, internally map the strings to integers, and use these integers to make splits - either via ordinal nature when nbins_cats is too small to resolve all levels or via bitsets that do a perfect group split. Each category is a separate category; its name (or number) is irrelevant. For example, after the strings are mapped to integers for Enum, you can split {0, 1, 2, 3, 4, 5} as {0, 4, 5} and {1, 2, 3}.
        enum_limited or EnumLimited: Automatically reduce categorical levels to the most prevalent ones during Aggregator training and only keep the T (1024) most frequent levels.
        one_hot_explicit or OneHotExplicit: N+1 new columns for categorical features with N levels
        binary or Binary: No more than 32 columns per categorical feature
        eigen or Eigen: k columns per categorical feature, keeping projections of one-hot-encoded matrix onto k-dim eigen space only
        label_encoder or LabelEncoder: Convert every enum into the integer of its index (for example, level 0 -> 0, level 1 -> 1, etc.) The categories are lexicographically mapped to numbers and lose their categorical nature, becoming ordinal. After the strings are mapped to integers, you can split {0, 1, 2, 3, 4, 5} as {0, 1, 2} and {3, 4, 5}.
        sort_by_response or SortByResponse: Reorders the levels by the mean response (for example, the level with lowest response -> 0, the level with second-lowest response -> 1, etc.). This is useful in GBM/DRF, for example, when you have more levels than nbins_cats, and where the top level splits now have a chance at separating the data with a split. Note that this requires a specified response column.

Deep Learning

        auto or AUTO: Allow the algorithm to decide. For Deep Learning, the algorithm will perform One Hot Internal encoding when auto is specified.
        one_hot_internal or OneHotInternal: Leave the dataset as is. This internally expands each row via one-hot encoding on the fly. (default)
        binary or Binary: No more than 32 columns per categorical feature
        eigen or Eigen: k columns per categorical feature, keeping projections of one-hot-encoded matrix onto k-dim eigen space only
        label_encoder or LabelEncoder: Convert every enum into the integer of its index (for example, level 0 -> 0, level 1 -> 1, etc.). The categories are lexicographically mapped to numbers and lose their categorical nature, becoming ordinal. After the strings are mapped to integers, you can split {0, 1, 2, 3, 4, 5} as {0, 1, 2} and {3, 4, 5}. This is useful for keeping the number of columns small for XGBoost or DeepLearning, where the algorithm otherwise perform ExplicitOneHotEncoding.
        sort_by_response or SortByResponse: Reorders the levels by the mean response (for example, the level with lowest response -> 0, the level with second-lowest response -> 1, etc.). Note that this requires a specified response column.

    Note: For Deep Learning, this value defaults to one_hot_internal. Similarly, if auto is specified, then the algorithm performs one_hot_internal encoding.

Aggregator

        auto or AUTO: Allow the algorithm to decide. For Aggregator, the algorithm will perform One Hot Internal encoding when auto is specified.
        one_hot_internal or OneHotInternal: Leave the dataset as is. This internally expands each row via one-hot encoding on the fly. (default)
        binary or Binary: No more than 32 columns per categorical feature
        eigen or Eigen: k columns per categorical feature, keeping projections of one-hot-encoded matrix onto k-dim eigen space only
        label_encoder or LabelEncoder: Convert every enum into the integer of its index (for example, level 0 -> 0, level 1 -> 1, etc.). The categories are lexicographically mapped to numbers and lose their categorical nature, becoming ordinal. After the strings are mapped to integers, you can split {0, 1, 2, 3, 4, 5} as {0, 1, 2} and {3, 4, 5}. This is useful for keeping the number of columns small.
        enum_limited or EnumLimited: Automatically reduce categorical levels to the most prevalent ones during Aggregator training and only keep the T (1024) most frequent levels.

XGBoost

        auto or AUTO: Allow the algorithm to decide (default). In XGBoost, the algorithm will automatically perform one_hot_internal encoding. (default)
        enum or Enum: 1 column per categorical feature. Each category is a separate category; its name (or number) is irrelevant. For example, after the strings are mapped to integers for Enum, you can split {0, 1, 2, 3, 4, 5} as {0, 4, 5} and {1, 2, 3}.
        one_hot_internal or OneHotInternal: On the fly N+1 new cols for categorical features with N levels
        one_hot_explicit or OneHotExplicit: N+1 new columns for categorical features with N levels
        binary: No more than 32 columns per categorical feature
        eigen or Eigen: k columns per categorical feature, keeping projections of one-hot-encoded matrix onto k-dim eigen space only
        label_encoder or LabelEncoder: Convert every enum into the integer of its index (for example, level 0 -> 0, level 1 -> 1, etc.) The categories are lexicographically mapped to numbers and lose their categorical nature, becoming ordinal. After the strings are mapped to integers, you can split {0, 1, 2, 3, 4, 5} as {0, 1, 2} and {3, 4, 5}.
        sort_by_response or SortByResponse: Reorders the levels by the mean response (for example, the level with lowest response -> 0, the level with second-lowest response -> 1, etc.). This is useful, for example, when you have more levels than nbins_cats, and where the top level splits now have a chance at separating the data with a split. Note that this requires a specified response column.
        enum_limited or EnumLimited: Automatically reduce categorical levels to the most prevalent ones during training and only keep the T (1024) most frequent levels.

K-Means

        auto or AUTO: Allow the algorithm to decide (default). For K-Means, the algorithm will perform Enum encoding when auto option is specified.
        enum or Enum: Leave the dataset as is, internally map the strings to integers, and use these integers to make splits - either via ordinal nature when nbins_cats is too small to resolve all levels or via bitsets that do a perfect group split. Each category is a separate category; its name (or number) is irrelevant. For example, after the strings are mapped to integers for Enum, you can split {0, 1, 2, 3, 4, 5} as {0, 4, 5} and {1, 2, 3}.
        one_hot_explicit or OneHotExplicit: N+1 new columns for categorical features with N levels
        binary or Binary: No more than 32 columns per categorical feature
        eigen or Eigen: k columns per categorical feature, keeping projections of one-hot-encoded matrix onto k-dim eigen space only
        label_encoder or LabelEncoder: Convert every enum into the integer of its index (for example, level 0 -> 0, level 1 -> 1, etc.) The categories are lexicographically mapped to numbers and lose their categorical nature, becoming ordinal. After the strings are mapped to integers, you can split {0, 1, 2, 3, 4, 5} as {0, 1, 2} and {3, 4, 5}.

########################################################################################
########################################################################################
########################################################################################
histogram_type:

By default (AUTO) GBM/DRF bins from min…max in steps of (max-min)/N. Use this option to specify the type of histogram to use for finding optimal split points. Available types include:

    AUTO
    UniformAdaptive
    Random
    QuantilesGlobal
    RoundRobin

When histogram_type="UniformAdaptive" is specified, each feature is binned into buckets of equal step size (not population). This is the fastest method, and usually performs well, but can lead to less accurate splits if the distribution is highly skewed.

H2O supports extremely randomized trees (XRT) via histogram_type="Random". When this is specified, the algorithm will sample N-1 points from min…max and use the sorted list of those to find the best split. The cut points are random rather than uniform. For example, to generate 4 bins for some feature ranging from 0-100, 3 random numbers would be generated in this range (13.2, 89.12, 45.0). The sorted list of these random numbers forms the histogram bin boundaries e.g. (0-13.2, 13.2-45.0, 45.0-89.12, 89.12-100).

When histogram_type="QuantilesGlobal" is specified, the feature distribution is taken into account with a quantile-based binning (where buckets have equal population). This computes nbins quantiles for each numeric (non-binary) column, then refines/pads each bucket (between two quantiles) uniformly (and randomly for remainders) into a total of nbins_top_level bins. This set of split points is then used for all levels of the tree: each leaf node histogram gets min/max-range adjusted (based on its population range) and also linearly refined/padded to end up with exactly nbins (level) bins to pick the best split from. For integer columns where this ends up with more than the unique number of distinct values, the algorithm falls back to the pure-integer buckets.

When histogram_type="RoundRobin" is specified, the algorithm will cycle through all histogram types (one per tree).
########################################################################################
########################################################################################
########################################################################################
nbins_cats: (GBM, DRF)

When the training data contains columns with categorical levels (factors), these factors are split by assigning an integer to each distinct categorical level, then binning the ordered integers according to the user-specified number of bins (which defaults to 1024 bins), and then picking the optimal split point among the bins. For example, if you have levels A,B,C,D,E,F,G at a certain node to be split, and you specify nbins_cats=4, then the buckets {A,B},{C,D},{E,F},{G} define the grouping during the first split. Only during the next split of {A,B} (down the tree) will GBM separate {A} and {B}.

########################################################################################
########################################################################################
########################################################################################
class_sampling_factors: (GBM, DRF, Deep Learning, Naïve-Bayes, AutoML)


By default, sampling factors will be automatically computed to obtain class balance during training. You can change this behavior using the class_sampling_factors option. This option sets an over/under-sampling ratio for each class (in lexicographic order). Note that this requires balance_classes=true.

########################################################################################
########################################################################################
########################################################################################
col_sample_rate: (GBM, XGBoost)
	Row and column sampling (sample_rate and col_sample_rate) can improve generalization and lead to lower validation and test set errors. Good general values for large datasets are around 0.7 to 0.8 (sampling 70-80 percent of the data) for both parameters. Column sampling per tree (col_sample_rate_per_tree) can also be used. Note that col_sample_rate_per_tree is multiplicative with col_sample_rate, so setting both parameters to 0.8, for example, results in 64% of columns being considered at any given node to split.


########################################################################################
########################################################################################
########################################################################################
distribution: (GBM, Deep Learning, XGBoost)

Description

Unlike in GLM, where users specify both a distribution family and a link for the loss function, in GBM, Deep Learning, and XGBoost, distributions and loss functions are tightly coupled. In these algorithms, a loss function is specified using the distribution parameter. When specifying the distribution, the loss function is automatically selected as well. For exponential families (such as Poisson, Gamma, and Tweedie), the canonical logarithmic link function is used.

By default, the loss function method performs AUTO distribution. In this case, the algorithm will guess the model type based on the response column type (specified using y). More specifically, if the response column type is numeric, AUTO defaults to “gaussian”; if categorical, AUTO defaults to bernoulli or multinomial depending on the number of response categories.

Certain cases can exist, however, in which the median starting value for this loss function can lead to poor results (for example, if the median is the lowest or highest value in a tree node). The distribution option allows you to specify a different method. Available methods include AUTO, bernoulli, multinomial, gaussian, poisson, gamma, laplace, quantile, huber, and tweedie.

    If the distribution is bernoulli, the response column must be 2-class categorical.
    If the distribution is quasibinomial, the response column must be numeric and binary. (Available in GBM only.)
    If the distribution is multinomial, the response column must be categorical.
    If the distribution is gaussian, the response column must be numeric.
    If the distribution is poisson, the response column must be numeric.
    If the distribution is gamma, the response column must be numeric.
    If the distribution is laplace, the response column must be numeric.
    If the distribution is quantile, the response column must be numeric.
    If the distribution is huber, the response column must be numeric.
    If the distribution is tweedie, the response column must be numeric.

NOTE: laplace, quantile, and huber are NOT available in XGBoost.

The following general guidelines apply when selecting a distribution:

    For Classification problems:

        Bernoulli and Quasibinomial distributions are used for binary outcomes.
        A Multinomial distribution can handle multiple discrete outcomes.

    For Regression problems:

        A Gaussian distribution is the function for continuous targets.
        A Poisson distribution is used for estimating counts.
        A Gamma distribution is used for estimating total values (such as claim payouts, rainfall, etc.).
        A Tweedie distribution is used for estimating densities.
        A Laplacian loss function (absolute L1-loss function) can predict the median percentile.
        A Quantile regression loss function can predict a specified percentile.
        A Huber loss function, a combination of squared error and absolute error, is more robust to outliers than L2 squared-loss function.

When quasibinomial is specified, the response must be numeric and binary. The response must also have a low value of 0 (negative class). Note that this option is available in GBM only.

When tweedie is specified, users must also specify a tweedie_power value. Users can tune over this option with values > 1.0 and < 2.0. More information is available here.

When quantile is specified, then users can also specify a quantile_alpha value, which defines the desired quantile when performing quantile regression. For example, if you want to predict the 80th percentile of a column’s value, then you can specify quantile_alpha=0.8. The quantile_alpha value defaults to 0.5 (i.e., the median value, and essentially the same as specifying distribution=laplace). Note that this option is not available in XGBoost.

When huber is specified, then users can also specify a huber_alpha value. This indicates the top percentile of error that should be considered as outliers. Note that this option is not available in XGBoost.

For all distributions except multinomial, you can specify an offset_column. Offsets are per-row “bias values” that are used during model training. For Gaussian distributions, they can be seen as simple corrections to the response (y) column. Instead of learning to predict the response (y-row), the model learns to predict the (row) offset of the response column. For other distributions, the offset corrections are applied in the linearized space before applying the inverse link function to get the actual response values. For more information, refer to the following link.

# ref: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/distribution.html


########################################################################################
huber_alpha: (GBM, Deep Learning)

huber_alpha parameter, which dictates the threshold between quadratic and linear loss (i.e. the top percentile of error that should be considered as outliers). This value must be between 0 and 1 and defaults to 0.9.


quantile_alpha: (GBM, Deep Learning)

The quantile_alpha parameter value defines the desired quantile when performing quantile regression. Used in combination with distribution = quantile, quantile_alpha activates the quantile loss function. For example, if you want to predict the 80th percentile of the response column’s value, then you can specify quantile_alpha=0.8. The quantile_alpha value defaults to 0.5 (i.e., the median value, essentially the same as specifying distribution=laplace).

oofset_column: It can be used in regression case, while we need to inject bias in the solution(which means that, y is shifted from the 0 or mean)


########################################################################################
########################################################################################
########################################################################################

impute_missing: (PCA)
interaction_pairs: (GLM)
	# define specific interaction pairs
	interaction_pairs = [("CRSDepTime", "UniqueCarrier"),
	                     ("CRSDepTime", "Origin"),
	                     ("UniqueCarrier", "Origin")]

interactions: (GLM) a list of interactions to a model



########################################################################################
########################################################################################
########################################################################################

learn_rate_annealing: (GBM) (danger, very danger)

Description

Use this option to reduce the learning rate by this factor after every tree. When used, then for N trees, you would start with learn_rate and end with learn_rate * learn_rate_annealing^N.

The following provides some reference factors. (Refer to Taylor series for more information.):

    0.99^100 = 0.366
    0.99^1000 = 4.3e-5
    0.999^1000 = 0.368
    0.999^10000 = 4.5e-5

With this option, then instead of learn_rate=0.01, you can try (for example) learn_rate=0.05 along with learn_rate_annealing=0.99. The result should converge much faster with almost the same accuracy. Note, however, that this can also result in overfitting, so use caution when specifying this option.




########################################################################################
########################################################################################
########################################################################################


max_abs_leafnode_pred: (GBM, XGBoost)

Description

When building a GBM model, this option reduces overfitting by limiting the maximum absolute value of a leaf node prediction. This option is mainly used for classification models. It is a pure regularization tuning parameter as it prevents any particular leaf node from making large absolute predictions, but it doesn’t directly relate to the actual final prediction (other than that the final value can’t be larger than ntrees * max_abs_leafnode_pred, by definition).

Usually- 2,3,5,7(very large dataset)


########################################################################################
########################################################################################
########################################################################################







########################################################################################
########################################################################################
########################################################################################









########################################################################################
########################################################################################
########################################################################################










########################################################################################
########################################################################################
########################################################################################







########################################################################################
########################################################################################
########################################################################################
