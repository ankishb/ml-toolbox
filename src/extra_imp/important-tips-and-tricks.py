very-intersting ==> https://www.kaggle.com/xaviermaxime/light-gbm-with-simple-engineered-features



https://www.dummies.com/programming/big-data/data-science/data-science-how-to-create-interactions-between-variables-with-python/

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
regression = LinearRegression(normalize=True)
crossvalidation = KFold(n=X.shape[0], n_folds=10, shuffle=True, random_state=1)


df = pd.DataFrame(X,columns=boston.feature_names)
baseline = np.mean(cross_val_score(regression, df, y, scoring=‘r2’, cv=crossvalidation,
 n_jobs=1))
interactions = list()
for feature_A in boston.feature_names:
 for feature_B in boston.feature_names:
  if feature_A > feature_B:
   df[‘interaction’] = df[feature_A] * df[feature_B]
   score = np.mean(cross_val_score(regression, df, y, scoring=‘r2’,
    cv=crossvalidation, n_jobs=1))
   if score > baseline:
    interactions.append((feature_A, feature_B, round(score,3)))
print ‘Baseline R2: %.3f’ % baseline
print ‘Top 10 interactions: %s’ % sorted(interactions, key=lambda(x):x[2],
 reverse=True)[:10]
Baseline R2: 0.699
Top 10 interactions: [(‘RM’, ‘LSTAT’, 0.782), (‘TAX’, ‘RM’, 0.766),
 (‘RM’, ‘RAD’, 0.759), (‘RM’, ‘PTRATIO’, 0.75), 
(‘RM’, ‘INDUS’, 0.748), (‘RM’, ‘NOX’, 0.733), 
(‘RM’, ‘B’, 0.731), (‘RM’, ‘AGE’, 0.727), 
(‘RM’, ‘DIS’, 0.722), (‘ZN’, ‘RM’, 0.716)]


polyX = pd.DataFrame(X,columns=boston.feature_names)
baseline = np.mean(cross_val_score(regression, polyX, y, 
scoring=‘mean_squared_error’,
 cv=crossvalidation, n_jobs=1))
improvements = [baseline]
for feature_A in boston.feature_names:
 polyX[feature_A+’^2’] = polyX[feature_A]**2
 improvements.append(np.mean(cross_val_score(regression, polyX, y,
  scoring=‘mean_squared_error’, cv=crossvalidation, n_jobs=1)))
 for feature_B in boston.feature_names:
  if feature_A > feature_B:
   polyX[feature_A+’*’+feature_B] = polyX[feature_A] * polyX[feature_B]
   improvements.append(np.mean(cross_val_score(regression, polyX, y,
    scoring=‘mean_squared_error’, cv=crossvalidation, n_jobs=1)))





1. Try out simple multiplication of cat column with data type as str

1.  1  1
2.  2  1
3.  1  2
4.  1  2
5.  2  2

Using the upper technique, new feature will be 

1.  1  1  1:1
2.  2  1  2:1
3.  1  2  1:2
4.  1  2  1:2
5.  2  2  2:2



==> Apply sigmoid/tanh function for tranformation
==> Binary variable, whether feature has null or not
==> check if is is fututre dataset in test
==> For faster experiment, use sampling method to select subset of training data, but don't touch validation
==> Always explore model/feature-engineering by subsetting the whole data

==> min_sample_leaf help in better generalization, choose 1,3,5,10,25 for data-set of range 100,000 sample, for bigger data-set tune this parameter to 100,1000 etc.
==> Choose max_feature to 0.5,sqrt,log for better generalization.

==> Check out distribution of danger column w.r.t each label 0/1



https://www.youtube.com/watch?v=42Oo8TOl85I
https://github.com/h2oai/h2o-tutorials/tree/master/h2o-world-2017/automl
H20 ==> feature-engineering  (https://github.com/h2oai/h2o-tutorials/blob/78c3766741e8cbbbd8db04d54b1e34f678b85310/best-practices/feature-engineering/feature_engineering.ipynb)

https://github.com/h2oai/h2o-3/blob/master/h2o-py/h2o/targetencoder.py




The issue is because you are trying encoding multiple categorical features. I think that is a bug of H2O, but you can solve putting the transformer in a for loop that iterate over all categorical names.

import numpy as np
import pandas as pd
import h2o
from h2o.targetencoder import TargetEncoder
h2o.init()

df = pd.DataFrame({
    'x_0': ['a'] * 5 + ['b'] * 5,
    'x_1': ['c'] * 9 + ['d'] * 1,
    'x_2': ['a'] * 3 + ['b'] * 7,
    'y_0': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
})

hf = h2o.H2OFrame(df)
hf['cv_fold_te'] = hf.kfold_column(n_folds=2, seed=54321)
hf['y_0'] = hf['y_0'].asfactor()
cat_features = ['x_0', 'x_1', 'x_2']

for item in cat_features:
    target_encoder = TargetEncoder(x=[item], y='y_0', fold_column = 'cv_fold_te')
    target_encoder.fit(hf)
    hf = target_encoder.transform(frame=hf, holdout_type='kfold',
                                  seed=54321, noise=0.0)
hf










==> Use boolean dtype for the category varaible True/False
==> Use profiling, to check which is taking too much time, as %prun clf.fit(X,y)
	But remember, it take complete data-set, instead of sampling method(Ambiguous, it may be case for fast.ai)
==> Create Lag feature, but think how (There are several IDs as State, Pincode, Employee, Branch, etc)
==> Create date-based features, for example, percentage change in sales between jan-feb, etc.
==> Create scatter plot of validation score and test score, it should have straight line between them, if it is not the case, then something weird is going on in your model.





1. My hypothesis that: if we use small no of tree, in gbm or random forest, their feature importance will be same.
2. If this is true, then intraction based feature can be used in gbm paradim to select the best feature out of that.
3. Subsemble approach, subset of original data to build model and combine final prediction in simple blending fashion.
	- Try out this hypothesis on hike dataset.
4. Run SVD on dense dataset, by replacing outliers with the nan and fit model on top of that.
5. Add permuattion importance function in your library
6. chi2 feature selevction test
7. Splitting of output in subset, shouldnot the splitting by binning is better?

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())




# ml-toolbox

######################################### Feature to add in your module #########################################
#################################################################################################################
#################################################################################################################

1. Add EDA tool for classification using hue.
	- boxplot
	- kdeplot
	- distplot
	- pairplot
	- multivariate plot
	- heatmap
		https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners

2. Common feature
  - Outliers Handling
    - filling with nan
    - removing them
    - fill with lower/upper bound 
    
6. Feature diversity specially for catgorical variable such as 
	- eigen-decomoposition
	- one-hot
	- target-encoding (cv based your and h2o)
		https://maxhalford.github.io/blog/target-encoding-done-the-right-way/
		http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/target-encoding.html
	- bayesian encoding (prior information) 
		https://github.com/MaxHalford/xam/blob/master/docs/feature-extraction.md#smooth-target-encoding
	- quantile binning
	- Label encoder
	- Hash encoder

7. Feature-engineering:
	- intraction based w/wo RF and boosting method
	- date-time based features
	- lag feature(time series) (grouping based on previous day/hour sales or sth)
	- relational (grouping based)
	- Matrix factorization/low-rank
	- social network (networkx)
	- Count-vectorizer
	- Hashing Tricks for highly sparse data(text data)
	- text preprocessing
	- rounding 
	- split decimal value 
		for example sales prices is 899.99, 500.01, this .99, .01 can be feature
	- non-linear dimensionality reduction (PCA, Kernel-PCA, tsne, svd, NMF)
  - outliers handling using matrix factorization




8. Deep learning module:
	- Augmentatioan function
	- custom callbacks
	- data on fly 
	- pretrained model (classes as_ in object detection)
	- simple layer in functional form


9. Advanced features
  - feature importance toolbox, eli5/shap-value
  - gridsearch / bayesian optimization
  - psuedo labeling
  - object detection
  


ML-Model:
	- XGBoost/LightGBM/CatBoost
	- ExtraTree/ Adaptive GBM/ Random-Forest
	- Linear model/ Lasso/Ridge/Logistic/SVM
	- KNN(coursera)/tsne-multicore/clustering
	- H2o models with all important parameters specifically with categorical_encoding
	- Non negative linear regression (scipy)/ lasso(positive=True)


Special features:
    - Parallel processing
    - Dask tutorial for data preprocessing
    - Feature Selection (Recusive feature elimination)
    - Sparse Matrix handling and along with Sparse SVD
    - OOF-analysis(correlation plot and analysis)
    - Error Analysis
    - CV Vs leaderboard analysis
    - Rank Average



Useful resources:
1. https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/select-features
2. https://forums.aws.amazon.com/message.jspa?messageID=774050
3. https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.nnls.html
4. http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/categorical_encoding.html
5. http://h2o-release.s3.amazonaws.com/h2o/master/3484/docs-website/h2o-py/docs/intro.html
6. https://christophm.github.io/interpretable-ml-book/extend-lm.html (book)
7. https://maxhalford.github.io/blog/streaming-groupbys-in-pandas-for-big-datasets/
8. https://web.stanford.edu/~hastie/ElemStatLearn/
9. https://www.dummies.com/programming/big-data/data-science/data-science-how-to-create-interactions-between-variables-with-python/


#################################################################################################################
#################################################################################################################
#################################################################################################################






# ##Unsupervised Anomaly detection
# For instructions on how to build unsupervised models with H2O Deep Learning, we refer to our previous [Tutorial on Anomaly Detection with H2O Deep Learning](https://www.youtube.com/watch?v=fUSbljByXak) and our [MNIST Anomaly detection code example](https://github.com/h2oai/h2o-3/blob/master/h2o-r/tests/testdir_algos/deeplearning/runit_deeplearning_anomaly_large.R), as well as our [Stacked AutoEncoder R code example](https://github.com/h2oai/h2o-3/blob/master/h2o-r/tests/testdir_algos/deeplearning/runit_deeplearning_stacked_autoencoder_large.R).
# 
# 
# ##H2O Deep Learning Tips & Tricks
# ####Activation Functions
# While sigmoids have been used historically for neural networks, H2O Deep Learning implements `Tanh`, a scaled and shifted variant of the sigmoid which is symmetric around 0. Since its output values are bounded by -1..1, the stability of the neural network is rarely endangered. However, the derivative of the tanh function is always non-zero and back-propagation (training) of the weights is more computationally expensive than for rectified linear units, or `Rectifier`, which is `max(0,x)` and has vanishing gradient for `x<=0`, leading to much faster training speed for large networks and is often the fastest path to accuracy on larger problems. In case you encounter instabilities with the `Rectifier` (in which case model building is automatically aborted), try a limited value to re-scale the weights: `max_w2=10`. The `Maxout` activation function is computationally more expensive, but can lead to higher accuracy. It is a generalized version of the Rectifier with two non-zero channels. In practice, the `Rectifier` (and `RectifierWithDropout`, see below) is the most versatile and performant option for most problems.
# 
# ####Generalization Techniques
# L1 and L2 penalties can be applied by specifying the `l1` and `l2` parameters. Intuition: L1 lets only strong weights survive (constant pulling force towards zero), while L2 prevents any single weight from getting too big. [Dropout](http://arxiv.org/pdf/1207.0580.pdf) has recently been introduced as a powerful generalization technique, and is available as a parameter per layer, including the input layer. `input_dropout_ratio` controls the amount of input layer neurons that are randomly dropped (set to zero), while `hidden_dropout_ratios` are specified for each hidden layer. The former controls overfitting with respect to the input data (useful for high-dimensional noisy data), while the latter controls overfitting of the learned features. Note that `hidden_dropout_ratios` require the activation function to end with `...WithDropout`.
# 
# ####Early stopping and optimizing for lowest validation error
# By default, Deep Learning training stops when the `stopping_metric` does not improve by at least `stopping_tolerance` (0.01 means 1% improvement) for `stopping_rounds` consecutive scoring events on the training (or validation) data. By default, `overwrite_with_best_model` is enabled and the model returned after training for the specified number of epochs (or after stopping early due to convergence) is the model that has the best training set error (according to the metric specified by `stopping_metric`), or, if a validation set is provided, the lowest validation set error. Note that the training or validation set errors can be based on a subset of the training or validation data, depending on the values for `score_validation_samples` or `score_training_samples`, see below. For early stopping on a predefined error rate on the *training data* (accuracy for classification or MSE for regression), specify `classification_stop` or `regression_stop`.
# 
# ####Training Samples per MapReduce Iteration
# The parameter `train_samples_per_iteration` matters especially in multi-node operation. It controls the number of rows trained on for each MapReduce iteration. Depending on the value selected, one MapReduce pass can sample observations, and multiple such passes are needed to train for one epoch. All H2O compute nodes then communicate to agree on the best model coefficients (weights/biases) so far, and the model may then be scored (controlled by other parameters below). The default value of `-2` indicates auto-tuning, which attemps to keep the communication overhead at 5% of the total runtime. The parameter `target_ratio_comm_to_comp` controls this ratio. This parameter is explained in more detail in the [H2O Deep Learning booklet](http://h2o.ai/resources/),
# 
# ####Categorical Data
# For categorical data, a feature with K factor levels is automatically one-hot encoded (horizontalized) into K-1 input neurons. Hence, the input neuron layer can grow substantially for datasets with high factor counts. In these cases, it might make sense to reduce the number of hidden neurons in the first hidden layer, such that large numbers of factor levels can be handled. In the limit of 1 neuron in the first hidden layer, the resulting model is similar to logistic regression with stochastic gradient descent, except that for classification problems, there's still a softmax output layer, and that the activation function is not necessarily a sigmoid (`Tanh`). If variable importances are computed, it is recommended to turn on `use_all_factor_levels` (K input neurons for K levels). The experimental option `max_categorical_features` uses feature hashing to reduce the number of input neurons via the hash trick at the expense of hash collisions and reduced accuracy. Another way to reduce the dimensionality of the (categorical) features is to use `h2o.glrm()`, we refer to the GLRM tutorial for more details.
# 
# ####Missing Values
# H2O Deep Learning automatically does mean imputation for missing values during training (leaving the input layer activation at 0 after standardizing the values). For testing, missing test set values are also treated the same way by default. See the `h2o.impute` function to do your own mean imputation.
# 
# ####Loss functions, Distributions, Offsets, Observation Weights
# H2O Deep Learning supports advanced statistical features such as multiple loss functions, non-Gaussian distributions, per-row offsets and observation weights.
# In addition to `Gaussian` distributions and `Squared` loss, H2O Deep Learning supports `Poisson`, `Gamma`, `Tweedie` and `Laplace` distributions. It also supports `Absolute` and `Huber` loss and per-row offsets specified via an `offset_column`. Observation weights are supported via a user-specified `weights_column`.
# 
# We refer to our [H2O Deep Learning R test code examples](https://github.com/h2oai/h2o-3/tree/master/h2o-r/tests/testdir_algos/deeplearning) for more information.
# a
# 
# ####Reproducibility
# Every run of DeepLearning results in different results since multithreading is done via [Hogwild!](http://www.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf) that benefits from intentional lock-free race conditions between threads. To get reproducible results for small datasets and testing purposes, set `reproducible=T` and set `seed=1337` (pick any integer). This will not work for big data for technical reasons, and is probably also not desired because of the significant slowdown (runs on 1 core only).
#     
# ####Scoring on Training/Validation Sets During Training  
# The training and/or validation set errors *can* be based on a subset of the training or validation data, depending on the values for `score_validation_samples` (defaults to 0: all) or `score_training_samples` (defaults to 10,000 rows, since the training error is only used for early stopping and monitoring). For large datasets, Deep Learning can automatically sample the validation set to avoid spending too much time in scoring during training, especially since scoring results are not currently displayed in the model returned to R.
#                                 
# Note that the default value of `score_duty_cycle=0.1` limits the amount of time spent in scoring to 10%, so a large number of scoring samples won't slow down overall training progress too much, but it will always score once after the first MapReduce iteration, and once at the end of training.
# 
# Stratified sampling of the validation dataset can help with scoring on datasets with class imbalance.  Note that this option also requires `balance_classes` to be enabled (used to over/under-sample the training dataset, based on the max. relative size of the resulting training dataset, `max_after_balance_size`):
#     
# ### More information can be found in the [H2O Deep Learning booklet](http://h2o.ai/resources/), in our [H2O SlideShare Presentations](http://www.slideshare.net/0xdata/presentations), our [H2O YouTube channel](https://www.youtube.com/user/0xdata/), as well as on our [H2O Github Repository](https://github.com/h2oai/h2o-3/), especially in our [H2O Deep Learning R tests](https://github.com/h2oai/h2o-3/tree/master/h2o-r/tests/testdir_algos/deeplearning), and [H2O Deep Learning Python tests](https://github.com/h2oai/h2o-3/tree/master/h2o-py/tests/testdir_algos/deeplearning).

# ###Further Exploration
# Due to the limited scope of this talk, only a portion of the code has been ported to Python.  
# 
# Additional topics are covered in R: Please see the deeplearning.R file for code samples for the following:
# 
# * Hyper-Parameter tuning with Grid Search  
# * Random Hyper-Parameter Search  
# * Checkpointing  
# * Cross-Validation  
# * Regression and Binary Classification  
# * Exporting Weights and Biases
# 
