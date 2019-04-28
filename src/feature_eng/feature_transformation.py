

import gc
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import PolynomialFeatures as pf



def get_polynomial_feature(train, test, cols, degree, interact, bias=False, return_fun=False):
    """ polynomial feature
    Args:
        train_df, test_df
        degree: degree of polynomial feature
        cols: columns to use
        interact: if True, return [x1, x2, x1*x2], else return [x1, x2, x1**2, x2**2, x1*x2]
        bias: if True, add column of ones as [1, x1, x2, x1*x2], else do nothing
        
    return:
        data-frame which extended feature obtained from up function
    example:
        X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1]])
		X = pd.DataFrame(data=X, columns=['x1','x2', 'x3'])
		new, new1 = get_polynomial_feature(X, X, cols=['x1', 'x2'], degree=2, interact=False)
    """
    from sklearn.preprocessing import PolynomialFeatures as pf
    complete_df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    if len(cols) == 1:
        raise ValueError('columns should be more than 1.')
        
    print("intraction happens only in cols: ", cols)
    interact_cols = cols
    no_interact_cols = list(set(train.columns) - set(cols))
    
    poly_feat = pf(degree=degree, interaction_only=interact, include_bias=bias)
    new_feat = poly_feat.fit_transform(complete_df[interact_cols])
    
    new_feat = pd.DataFrame(data=new_feat, columns=poly_feat.get_feature_names())
    new_feat = pd.concat([complete_df[no_interact_cols], new_feat], axis=1)
    
    train = new_feat.iloc[:train.shape[0]]
    test  = new_feat.iloc[train.shape[0]:].reset_index(drop=True)
    
    del complete_df, new_feat
    gc.collect()
    if return_fun == True:
        return train, test, poly_feat
    else:
        return train, test
    
    

def nmf_decomposition(train, test, n_component, alpha=0, l1_ratio=0, col_name=None):
	"""return nmf transformation
	Args:
		train, test: dataframe
		col_name: list of column name to be transform [if None, used all column]
		n_component: no of component to be used
	return:
		Transformed feature space
	example:
		train_nmf, test_nmf, nmf = nmf_decomposition(X.iloc[:10], X.iloc[10:15], n_component=2)
		train_nmf, test_nmf, nmf = nmf_decomposition(X.iloc[:10], X.iloc[10:15], n_component=2, alpha=0.1, l1_ratio=0.2)
	"""
	if col_name is None:
		col_name = train.columns

	complete_df = pd.concat([train[col_name], test[col_name]], axis=0)

	nmf = NMF(n_components=None, random_state=1234, alpha=alpha, l1_ratio=l1_ratio)
	nmf.fit(complete_df)
	complete_nmf = nmf.transform(complete_df)

	complete_nmf = pd.DataFrame(data=complete_nmf)
	complete_nmf.columns = ['nmf_'+str(i) for i in range(n_component)]

	train_nmf = complete_nmf.iloc[:train.shape[0]]
	test_nmf = complete_nmf.iloc[train.shape[0]:].reset_index(drop=True)

	del complete_nmf, complete_df
	gc.collect()
	return train_nmf, test_nmf, nmf


def svd_decomposition(train, test, n_component, col_name=None):
	"""return svd transformation
	Args:
		train, test: dataframe
		col_name: list of column name to be transform [if None, used all column]
		n_component: no of component to be used
	example:
		train_svd, test_svd, svd = svd_decomposition(X.iloc[:10], X.iloc[10:15], 2)
	"""
	if col_name is None:
		col_name = train.columns

	complete_df = pd.concat([train[col_name], test[col_name]], axis=0)

	svd = TruncatedSVD(n_components=n_component)
	svd.fit(complete_df)
	complete_svd = svd.transform(complete_df)

	complete_svd = pd.DataFrame(data=complete_svd)
	complete_svd.columns = ['svd_'+str(i) for i in range(n_component)]

	train_svd = complete_svd.iloc[:train.shape[0]]
	test_svd = complete_svd.iloc[train.shape[0]:].reset_index(drop=True)

	del complete_svd, complete_df
	gc.collect()
	return train_svd, test_svd, svd



# The objective function is:

# 0.5 * ||X - WH||_Fro^2
# + alpha * l1_ratio * ||vec(W)||_1
# + alpha * l1_ratio * ||vec(H)||_1
# + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
# + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2

# Where:

# ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
# ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)





# from sklearn.preprocessing import KBinsDiscretizer
# kbins = KBinsDiscretizer(n_bins=n_bins, encode=encoding, 
# 	strategy=strategy)

# encode:  {‘onehot’, ‘onehot-dense’, ‘ordinal’}
# strategy : {‘uniform’, ‘quantile’, ‘kmeans’}

# return: bin_edges_

# X = [[-2, 1, -4,   -1],
#      [-1, 2, -3, -0.5],
#      [ 0, 3, -2,  0.5],
#      [ 1, 4, -1,    2]]
# est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
# est.fit(X)  






# import numpy as np
# from sklearn.preprocessing import FunctionTransformer
# transformer = FunctionTransformer(np.log1p, validate=True)
# X = np.array([[0, 1], [2, 3]])
# transformer.transform(X)

# Xt = est.transform(X)
# Xt













##########################################################################################
##########################################################################################
 class sklearn.decomposition.FastICA(n_components=None, algorithm=’parallel’, whiten=True, fun=’logcosh’, fun_args=None, max_iter=200, tol=0.0001, w_init=None, random_state=None)[source]

    FastICA: a fast algorithm for Independent Component Analysis.

    Read more in the User Guide.
    Parameters:	

    n_components : int, optional

        Number of components to use. If none is passed, all are used.
    algorithm : {‘parallel’, ‘deflation’}

        Apply parallel or deflational algorithm for FastICA.
    whiten : boolean, optional

        If whiten is false, the data is already considered to be whitened, and no whitening is performed.
    fun : string or function, optional. Default: ‘logcosh’

        The functional form of the G function used in the approximation to neg-entropy. Could be either ‘logcosh’, ‘exp’, or ‘cube’. You can also provide your own function. It should return a tuple containing the value of the function, and of its derivative, in the point. Example:

        def my_g(x):

            return x ** 3, (3 * x ** 2).mean(axis=-1)

    fun_args : dictionary, optional

        Arguments to send to the functional form. If empty and if fun=’logcosh’, fun_args will take value {‘alpha’ : 1.0}.
    max_iter : int, optional

        Maximum number of iterations during fit.
    tol : float, optional

        Tolerance on update at each iteration.
    w_init : None of an (n_components, n_components) ndarray

        The mixing matrix to be used to initialize the algorithm.
    random_state : int, RandomState instance or None, optional (default=None)

        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

    Attributes:	

    components_ : 2D array, shape (n_components, n_features)

        The unmixing matrix.
    mixing_ : array, shape (n_features, n_components)

        The mixing matrix.
    n_iter_ : int

        If the algorithm is “deflation”, n_iter is the maximum number of iterations run across all components. Else they are just the number of iterations taken to converge.

    Notes

    Implementation based on A. Hyvarinen and E. Oja, Independent Component Analysis: Algorithms and Applications, Neural Networks, 13(4-5), 2000, pp. 411-430

    Examples
    >>>

    from sklearn.datasets import load_digits
    from sklearn.decomposition import FastICA
    X, _ = load_digits(return_X_y=True)
    transformer = FastICA(n_components=7,
            random_state=0)
    X_transformed = transformer.fit_transform(X)
    X_transformed.shape

##########################################################################################
##########################################################################################



##########################################################################################
##########################################################################################
 class sklearn.decomposition.IncrementalPCA(n_components=None, whiten=False, copy=True, batch_size=None)[source]¶

    Incremental principal components analysis (IPCA).

    Linear dimensionality reduction using Singular Value Decomposition of centered data, keeping only the most significant singular vectors to project the data to a lower dimensional space.

    Depending on the size of the input data, this algorithm can be much more memory efficient than a PCA.

    This algorithm has constant memory complexity, on the order of batch_size, enabling use of np.memmap files without loading the entire file into memory.

    The computational overhead of each SVD is O(batch_size * n_features ** 2), but only 2 * batch_size samples remain in memory at a time. There will be n_samples / batch_size SVD computations to get the principal components, versus 1 large SVD of complexity O(n_samples * n_features ** 2) for PCA.

    Read more in the User Guide.
    Parameters:	

    n_components : int or None, (default=None)

        Number of components to keep. If n_components `` is ``None, then n_components is set to min(n_samples, n_features).
    whiten : bool, optional

        When True (False by default) the components_ vectors are divided by n_samples times components_ to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal (the relative variance scales of the components) but can sometimes improve the predictive accuracy of the downstream estimators by making data respect some hard-wired assumptions.
    copy : bool, (default=True)

        If False, X will be overwritten. copy=False can be used to save memory but is unsafe for general use.
    batch_size : int or None, (default=None)

        The number of samples to use for each batch. Only used when calling fit. If batch_size is None, then batch_size is inferred from the data and set to 5 * n_features, to provide a balance between approximation accuracy and memory consumption.




from sklearn.datasets import load_digits
from sklearn.decomposition import IncrementalPCA
X, _ = load_digits(return_X_y=True)
transformer = IncrementalPCA(n_components=7, batch_size=200)
# either partially fit on smaller batches of data
transformer.partial_fit(X[:100, :])

# or let the fit function itself divide the data into batches
X_transformed = transformer.fit_transform(X)
X_transformed.shape



##########################################################################################
##########################################################################################



##########################################################################################
##########################################################################################

 class sklearn.decomposition.LatentDirichletAllocation(n_components=10, doc_topic_prior=None, topic_word_prior=None, learning_method=’batch’, learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1, total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=None, n_topics=None)[source]

    Latent Dirichlet Allocation with online variational Bayes algorithm

    New in version 0.17.

    Read more in the User Guide.
    Parameters:	

    n_components : int, optional (default=10)

        Number of topics.
    doc_topic_prior : float, optional (default=None)

        Prior of document topic distribution theta. If the value is None, defaults to 1 / n_components. In [Re25e5648fc37-1], this is called alpha.
    topic_word_prior : float, optional (default=None)

        Prior of topic word distribution beta. If the value is None, defaults to 1 / n_components. In [Re25e5648fc37-1], this is called eta.
    learning_method : ‘batch’ | ‘online’, default=’batch’

        Method used to update _component. Only used in fit method. In general, if the data size is large, the online update will be much faster than the batch update.

        Valid options:

        'batch': Batch variational Bayes method. Use all training data in
            each EM update.
            Old `components_` will be overwritten in each iteration.
        'online': Online variational Bayes method. In each EM update, use
            mini-batch of training data to update the ``components_``
            variable incrementally. The learning rate is controlled by the
            ``learning_decay`` and the ``learning_offset`` parameters.

        Changed in version 0.20: The default learning method is now "batch".
    learning_decay : float, optional (default=0.7)

        It is a parameter that control learning rate in the online learning method. The value should be set between (0.5, 1.0] to guarantee asymptotic convergence. When the value is 0.0 and batch_size is n_samples, the update method is same as batch learning. In the literature, this is called kappa.
    learning_offset : float, optional (default=10.)

        A (positive) parameter that downweights early iterations in online learning. It should be greater than 1.0. In the literature, this is called tau_0.
    max_iter : integer, optional (default=10)

        The maximum number of iterations.
    batch_size : int, optional (default=128)

        Number of documents to use in each EM iteration. Only used in online learning.
    evaluate_every : int, optional (default=0)

        How often to evaluate perplexity. Only used in fit method. set it to 0 or negative number to not evalute perplexity in training at all. Evaluating perplexity can help you check convergence in training process, but it will also increase total training time. Evaluating perplexity in every iteration might increase training time up to two-fold.
    total_samples : int, optional (default=1e6)

        Total number of documents. Only used in the partial_fit method.
    perp_tol : float, optional (default=1e-1)

        Perplexity tolerance in batch learning. Only used when evaluate_every is greater than 0.
    mean_change_tol : float, optional (default=1e-3)

        Stopping tolerance for updating document topic distribution in E-step.
    max_doc_update_iter : int (default=100)

        Max number of iterations for updating document topic distribution in the E-step.
    n_jobs : int or None, optional (default=None)

        The number of jobs to use in the E-step. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
    verbose : int, optional (default=0)

        Verbosity level.
    random_state : int, RandomState instance or None, optional (default=None)

        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    n_topics : int, optional (default=None)

        This parameter has been renamed to n_components and will be removed in version 0.21. .. deprecated:: 0.19

    Attributes:	

    components_ : array, [n_components, n_features]

        Variational parameters for topic word distribution. Since the complete conditional for topic word distribution is a Dirichlet, components_[i, j] can be viewed as pseudocount that represents the number of times word j was assigned to topic i. It can also be viewed as distribution over the words for each topic after normalization: model.components_ / model.components_.sum(axis=1)[:, np.newaxis].
    n_batch_iter_ : int

        Number of iterations of the EM step.
    n_iter_ : int

        Number of passes over the dataset.

    References

    [1] “Online Learning for Latent Dirichlet Allocation”, Matthew D. Hoffman,
        David M. Blei, Francis Bach, 2010
    [2] “Stochastic Variational Inference”, Matthew D. Hoffman, David M. Blei,
        Chong Wang, John Paisley, 2013
    [3] Matthew D. Hoffman’s onlineldavb code. Link:
        https://github.com/blei-lab/onlineldavb

    Examples
    >>>

    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.datasets import make_multilabel_classification
    # This produces a feature matrix of token counts, similar to what
    # CountVectorizer would produce on text.
    X, _ = make_multilabel_classification(random_state=0)
    lda = LatentDirichletAllocation(n_components=5,
        random_state=0)
    lda.fit(X) 

    # get topics for some given samples:
    lda.transform(X[-2:]



##########################################################################################
##########################################################################################



##########################################################################################
##########################################################################################
 class sklearn.decomposition.KernelPCA(n_components=None, kernel=’linear’, gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver=’auto’, tol=0, max_iter=None, remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None)[source]

    Kernel Principal component analysis (KPCA)

    Non-linear dimensionality reduction through the use of kernels (see Pairwise metrics, Affinities and Kernels).

    Read more in the User Guide.
    Parameters:	

    n_components : int, default=None

        Number of components. If None, all non-zero components are kept.
    kernel : “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”

        Kernel. Default=”linear”.
    gamma : float, default=1/n_features

        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels.
    degree : int, default=3

        Degree for poly kernels. Ignored by other kernels.
    coef0 : float, default=1

        Independent term in poly and sigmoid kernels. Ignored by other kernels.
    kernel_params : mapping of string to any, default=None

        Parameters (keyword arguments) and values for kernel passed as callable object. Ignored by other kernels.
    alpha : int, default=1.0

        Hyperparameter of the ridge regression that learns the inverse transform (when fit_inverse_transform=True).
    fit_inverse_transform : bool, default=False

        Learn the inverse transform for non-precomputed kernels. (i.e. learn to find the pre-image of a point)
    eigen_solver : string [‘auto’|’dense’|’arpack’], default=’auto’

        Select eigensolver to use. If n_components is much less than the number of training samples, arpack may be more efficient than the dense eigensolver.
    tol : float, default=0

        Convergence tolerance for arpack. If 0, optimal value will be chosen by arpack.
    max_iter : int, default=None

        Maximum number of iterations for arpack. If None, optimal value will be chosen by arpack.
    remove_zero_eig : boolean, default=False

        If True, then all components with zero eigenvalues are removed, so that the number of components in the output may be < n_components (and sometimes even zero due to numerical instability). When n_components is None, this parameter is ignored and components with zero eigenvalues are removed regardless.
    random_state : int, RandomState instance or None, optional (default=None)

        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. Used when eigen_solver == ‘arpack’.

        New in version 0.18.
    copy_X : boolean, default=True

        If True, input X is copied and stored by the model in the X_fit_ attribute. If no further changes will be done to X, setting copy_X=False saves memory by storing a reference.

        New in version 0.18.
    n_jobs : int or None, optional (default=None)

        The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

        New in version 0.18.

    Attributes:	

    lambdas_ : array, (n_components,)

        Eigenvalues of the centered kernel matrix in decreasing order. If n_components and remove_zero_eig are not set, then all values are stored.
    alphas_ : array, (n_samples, n_components)

        Eigenvectors of the centered kernel matrix. If n_components and remove_zero_eig are not set, then all components are stored.
    dual_coef_ : array, (n_samples, n_features)

        Inverse transform matrix. Only available when fit_inverse_transform is True.
    X_transformed_fit_ : array, (n_samples, n_components)

        Projection of the fitted data on the kernel principal components. Only available when fit_inverse_transform is True.
    X_fit_ : (n_samples, n_features)

        The data used to fit the model. If copy_X=False, then X_fit_ is a reference. This attribute is used for the calls to transform.

    References

    Kernel PCA was introduced in:
        Bernhard Schoelkopf, Alexander J. Smola, and Klaus-Robert Mueller. 1999. Kernel principal component analysis. In Advances in kernel methods, MIT Press, Cambridge, MA, USA 327-352.

    Examples
    >>>

    from sklearn.datasets import load_digits
    from sklearn.decomposition import KernelPCA
    X, _ = load_digits(return_X_y=True)
    transformer = KernelPCA(n_components=7, kernel='linear')
    X_transformed = transformer.fit_transform(X)
    X_transformed.shap



##########################################################################################
##########################################################################################


##########################################################################################
##########################################################################################

 class sklearn.decomposition.FactorAnalysis(n_components=None, tol=0.01, copy=True, max_iter=1000, noise_variance_init=None, svd_method=’randomized’, iterated_power=3, random_state=0)[source]

    Factor Analysis (FA)

    A simple linear generative model with Gaussian latent variables.

    The observations are assumed to be caused by a linear transformation of lower dimensional latent factors and added Gaussian noise. Without loss of generality the factors are distributed according to a Gaussian with zero mean and unit covariance. The noise is also zero mean and has an arbitrary diagonal covariance matrix.

    If we would restrict the model further, by assuming that the Gaussian noise is even isotropic (all diagonal entries are the same) we would obtain PPCA.

    FactorAnalysis performs a maximum likelihood estimate of the so-called loading matrix, the transformation of the latent variables to the observed ones, using expectation-maximization (EM).

    Read more in the User Guide.
    Parameters:	

    n_components : int | None

        Dimensionality of latent space, the number of components of X that are obtained after transform. If None, n_components is set to the number of features.
    tol : float

        Stopping tolerance for EM algorithm.
    copy : bool

        Whether to make a copy of X. If False, the input X gets overwritten during fitting.
    max_iter : int

        Maximum number of iterations.
    noise_variance_init : None | array, shape=(n_features,)

        The initial guess of the noise variance for each feature. If None, it defaults to np.ones(n_features)
    svd_method : {‘lapack’, ‘randomized’}

        Which SVD method to use. If ‘lapack’ use standard SVD from scipy.linalg, if ‘randomized’ use fast randomized_svd function. Defaults to ‘randomized’. For most applications ‘randomized’ will be sufficiently precise while providing significant speed gains. Accuracy can also be improved by setting higher values for iterated_power. If this is not sufficient, for maximum precision you should choose ‘lapack’.
    iterated_power : int, optional

        Number of iterations for the power method. 3 by default. Only used if svd_method equals ‘randomized’
    random_state : int, RandomState instance or None, optional (default=0)

        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. Only used when svd_method equals ‘randomized’.

    Attributes:	

    components_ : array, [n_components, n_features]

        Components with maximum variance.
    loglike_ : list, [n_iterations]

        The log likelihood at each iteration.
    noise_variance_ : array, shape=(n_features,)

        The estimated noise variance for each feature.
    n_iter_ : int

        Number of iterations run.

    See also

    PCA
        Principal component analysis is also a latent linear variable model which however assumes equal noise variance for each feature. This extra assumption makes probabilistic PCA faster as it can be computed in closed form.
    FastICA
        Independent component analysis, a latent variable model with non-Gaussian latent variables.

    References

    Examples
    >>>

    from sklearn.datasets import load_digits
    from sklearn.decomposition import FactorAnalysis
    X, _ = load_digits(return_X_y=True)
    transformer = FactorAnalysis(n_components=7, random_state=0)
    X_transformed = transformer.fit_transform(X)
    X_transformed.shape



##########################################################################################
##########################################################################################


##########################################################################################
##########################################################################################



################################################# MANIFOLD #########################################
##########################################################################################


 class sklearn.manifold.Isomap(n_neighbors=5, n_components=2, eigen_solver=’auto’, tol=0, max_iter=None, path_method=’auto’, neighbors_algorithm=’auto’, n_jobs=None)[source]

    Isomap Embedding

    Non-linear dimensionality reduction through Isometric Mapping

    Read more in the User Guide.
    Parameters:	

    n_neighbors : integer

        number of neighbors to consider for each point.
    n_components : integer

        number of coordinates for the manifold
    eigen_solver : [‘auto’|’arpack’|’dense’]

        ‘auto’ : Attempt to choose the most efficient solver for the given problem.

        ‘arpack’ : Use Arnoldi decomposition to find the eigenvalues and eigenvectors.

        ‘dense’ : Use a direct solver (i.e. LAPACK) for the eigenvalue decomposition.
    tol : float

        Convergence tolerance passed to arpack or lobpcg. not used if eigen_solver == ‘dense’.
    max_iter : integer

        Maximum number of iterations for the arpack solver. not used if eigen_solver == ‘dense’.
    path_method : string [‘auto’|’FW’|’D’]

        Method to use in finding shortest path.

        ‘auto’ : attempt to choose the best algorithm automatically.

        ‘FW’ : Floyd-Warshall algorithm.

        ‘D’ : Dijkstra’s algorithm.
    neighbors_algorithm : string [‘auto’|’brute’|’kd_tree’|’ball_tree’]

        Algorithm to use for nearest neighbors search, passed to neighbors.NearestNeighbors instance.
    n_jobs : int or None, optional (default=None)

        The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

    Attributes:	

    embedding_ : array-like, shape (n_samples, n_components)

        Stores the embedding vectors.
    kernel_pca_ : object

        KernelPCA object used to implement the embedding.
    training_data_ : array-like, shape (n_samples, n_features)

        Stores the training data.
    nbrs_ : sklearn.neighbors.NearestNeighbors instance

        Stores nearest neighbors instance, including BallTree or KDtree if applicable.
    dist_matrix_ : array-like, shape (n_samples, n_samples)

        Stores the geodesic distance matrix of training data.

    References
    [1]	Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric framework for nonlinear dimensionality reduction. Science 290 (5500)

    Examples
    >>>

    from sklearn.datasets import load_digits
    from sklearn.manifold import Isomap
    X, _ = load_digits(return_X_y=True)
    X.shape

    embedding = Isomap(n_components=2)
    X_transformed = embedding.fit_transform(X[:100])
    X_transformed.shape
##########################################################################################
##########################################################################################



##########################################################################################
##########################################################################################
Use only ‘modified’ or ‘ltsa’, hessian is also better, but time consuming
 class sklearn.manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=2, reg=0.001, eigen_solver=’auto’, tol=1e-06, max_iter=100, method=’standard’, hessian_tol=0.0001, modified_tol=1e-12, neighbors_algorithm=’auto’, random_state=None, n_jobs=None)[source]

    Locally Linear Embedding

    Read more in the User Guide.
    Parameters:	

    n_neighbors : integer

        number of neighbors to consider for each point.
    n_components : integer

        number of coordinates for the manifold
    reg : float

        regularization constant, multiplies the trace of the local covariance matrix of the distances.
    eigen_solver : string, {‘auto’, ‘arpack’, ‘dense’}

        auto : algorithm will attempt to choose the best method for input data

        arpack : use arnoldi iteration in shift-invert mode.

            For this method, M may be a dense matrix, sparse matrix, or general linear operator. Warning: ARPACK can be unstable for some problems. It is best to try several random seeds in order to check results.
        dense : use standard dense matrix operations for the eigenvalue

            decomposition. For this method, M must be an array or matrix type. This method should be avoided for large problems.

    tol : float, optional

        Tolerance for ‘arpack’ method Not used if eigen_solver==’dense’.
    max_iter : integer

        maximum number of iterations for the arpack solver. Not used if eigen_solver==’dense’.
    method : string (‘standard’, ‘hessian’, ‘modified’ or ‘ltsa’)

        standard : use the standard locally linear embedding algorithm. see

            reference [1]
        hessian : use the Hessian eigenmap method. This method requires

            n_neighbors > n_components * (1 + (n_components + 1) / 2 see reference [2]
        modified : use the modified locally linear embedding algorithm.

            see reference [3]
        ltsa : use local tangent space alignment algorithm

            see reference [4]

    hessian_tol : float, optional

        Tolerance for Hessian eigenmapping method. Only used if method == 'hessian'
    modified_tol : float, optional

        Tolerance for modified LLE method. Only used if method == 'modified'
    neighbors_algorithm : string [‘auto’|’brute’|’kd_tree’|’ball_tree’]

        algorithm to use for nearest neighbors search, passed to neighbors.NearestNeighbors instance
    random_state : int, RandomState instance or None, optional (default=None)

        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. Used when eigen_solver == ‘arpack’.
    n_jobs : int or None, optional (default=None)

        The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

    Attributes:	

    embedding_ : array-like, shape [n_samples, n_components]

        Stores the embedding vectors
    reconstruction_error_ : float

        Reconstruction error associated with embedding_
    nbrs_ : NearestNeighbors object

        Stores nearest neighbors instance, including BallTree or KDtree if applicable.

    References
    [1]	Roweis, S. & Saul, L. Nonlinear dimensionality reduction by locally linear embedding. Science 290:2323 (2000).
    [2]	Donoho, D. & Grimes, C. Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data. Proc Natl Acad Sci U S A. 100:5591 (2003).
    [3]	Zhang, Z. & Wang, J. MLLE: Modified Locally Linear Embedding Using Multiple Weights. http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382
    [4]	Zhang, Z. & Zha, H. Principal manifolds and nonlinear dimensionality reduction via tangent space alignment. Journal of Shanghai Univ. 8:406 (2004)

    Examples
    >>>

    from sklearn.datasets import load_digits
    from sklearn.manifold import LocallyLinearEmbedding
    X, _ = load_digits(return_X_y=True)
    X.shape

    embedding = LocallyLinearEmbedding(n_components=2)
    X_transformed = embedding.fit_transform(X[:100])
    X_transformed.shape


##########################################################################################
##########################################################################################



##########################################################################################
##########################################################################################


sklearn.manifold.spectral_embedding(adjacency, n_components=8, eigen_solver=None, random_state=None, eigen_tol=0.0, norm_laplacian=True, drop_first=True)[source]¶

    Project the sample on the first eigenvectors of the graph Laplacian.

    The adjacency matrix is used to compute a normalized graph Laplacian whose spectrum (especially the eigenvectors associated to the smallest eigenvalues) has an interpretation in terms of minimal number of cuts necessary to split the graph into comparably sized components.

    This embedding can also ‘work’ even if the adjacency variable is not strictly the adjacency matrix of a graph but more generally an affinity or similarity matrix between samples (for instance the heat kernel of a euclidean distance matrix or a k-NN matrix).

    However care must taken to always make the affinity matrix symmetric so that the eigenvector decomposition works as expected.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Read more in the User Guide.
    Parameters:	

    adjacency : array-like or sparse matrix, shape: (n_samples, n_samples)

        The adjacency matrix of the graph to embed.
    n_components : integer, optional, default 8

        The dimension of the projection subspace.
    eigen_solver : {None, ‘arpack’, ‘lobpcg’, or ‘amg’}, default None

        The eigenvalue decomposition strategy to use. AMG requires pyamg to be installed. It can be faster on very large, sparse problems, but may also lead to instabilities.
    random_state : int, RandomState instance or None, optional, default: None

        A pseudo random number generator used for the initialization of the lobpcg eigenvectors decomposition. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. Used when solver == ‘amg’.
    eigen_tol : float, optional, default=0.0

        Stopping criterion for eigendecomposition of the Laplacian matrix when using arpack eigen_solver.
    norm_laplacian : bool, optional, default=True

        If True, then compute normalized Laplacian.
    drop_first : bool, optional, default=True

        Whether to drop the first eigenvector. For spectral embedding, this should be True as the first eigenvector should be constant vector for connected graph, but for spectral clustering, this should be kept as False to retain the first eigenvector.

    Returns:	

    embedding : array, shape=(n_samples, n_components)

        The reduced samples.

    Notes

    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph has one connected component. If there graph has many components, the first few eigenvectors will simply uncover the connected components of the graph.

    References

        https://en.wikipedia.org/wiki/LOBPCG
        Toward the Optimal Preconditioned Eigensolver: Locally Optimal Block Preconditioned Conjugate Gradient Method Andrew V. Knyazev https://doi.org/10.1137%2FS1064827500366124




se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)

##########################################################################################
##########################################################################################



##########################################################################################
##########################################################################################



##########################################################################################
##########################################################################################




##########################################################################################
##########################################################################################



##########################################################################################
##########################################################################################
