

import gc
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.preprocessing import PolynomialFeatures as pf



def get_polynomial_feature(train_df, test_df, cols, degree, interact, bias=False, return_fun=False):
    """ polynomial feature
    Args:
        train_df, test_df: train and test are pandas dataframes
        degree: degree of polynomial feature
        cols: columns to use
        interact: if True, return [x1, x2, x1*x2], else return [x1, x2, x1**2, x2**2, x1*x2]
        bias: if True, add column of ones as [1, x1, x2, x1*x2], otherwise skip
        
    return:
        data-frame which extended feature obtained from up function
    example:
        X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1]])
        X = pd.DataFrame(data=X, columns=['x1','x2', 'x3'])
        new, new1 = get_polynomial_feature(X, X, cols=['x1', 'x2'], degree=2, interact=False)
    """
    complete_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    if len(cols) == 1:
        raise ValueError("columns size should be more than 1, to obtain the interaction")
        
    print("Columns list of interaction: ", cols)
    interact_cols = cols
    no_interact_cols = list(set(train.columns) - set(cols))

    poly_feat = pf(degree=degree, interaction_only=interact, include_bias=bias)
    new_feat = poly_feat.fit_transform(complete_df[interact_cols])

    new_feat = pd.DataFrame(data=new_feat, columns=poly_feat.get_feature_names())
    new_feat = pd.concat([complete_df[no_interact_cols], new_feat], axis=1)

    train_pf = new_feat.iloc[:train.shape[0]]
    test_pf  = new_feat.iloc[train.shape[0]:].reset_index(drop=True)

    del complete_df, new_feat
    gc.collect()
    if return_fun == True:
        return train_pf, test_pf, poly_feat
    else:
        return train_pf, test_pf
    
    

def nmf_decomposition(train, test, n_component, alpha=0, l1_ratio=0, col_name=None):
    """return nmf transformation
    Args:
        train, test: dataframe
        col_name: list of column name to be transform [if None, used all column]
        n_component: no of component to be used
        alpha=0 : Constant that multiplies the regularization terms. Set it to zero to have no regularization.
        l1_ratio : ( 0 <= l1_ratio <= 1).
            For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius Norm). 
            For l1_ratio = 1 it is an elementwise L1 penalty.
    return:
        Transformed feature space
    example:
        train_nmf, test_nmf, nmf = nmf_decomposition(X.iloc[:10], X.iloc[10:15], n_component=2, alpha=0.1, l1_ratio=0.2)
    """
    if col_name is None:
        col_name = train.columns

    complete_df = pd.concat([train[col_name], test[col_name]], axis=0)

    nmf = NMF(n_components=None, random_state=1234, alpha=alpha, l1_ratio=l1_ratio)
    complete_nmf = nmf.fit_transform(complete_df)

    complete_nmf = pd.DataFrame(data=complete_nmf)
    complete_nmf.columns = ['nmf_'+str(i) for i in range(n_component)]

    train_nmf = complete_nmf.iloc[:train.shape[0]]
    test_nmf  = complete_nmf.iloc[train.shape[0]:].reset_index(drop=True)

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
    complete_svd = svd.fit_transform(complete_df)

    complete_svd = pd.DataFrame(data=complete_svd)
    complete_svd.columns = ['svd_'+str(i) for i in range(n_component)]

    train_svd = complete_svd.iloc[:train.shape[0]]
    test_svd = complete_svd.iloc[train.shape[0]:].reset_index(drop=True)

    del complete_svd, complete_df
    gc.collect()
    return train_svd, test_svd, svd








def get_decomposition(train, test=None, n_component=2, col_name=None, which_method='svd', alpha=0.5, l1_ratio=0.3):
    """return svd transformation
    Args:
        train, test: dataframe
        col_name: list of column name to be transform [if None, used all column]
        n_component: no of component to be used
    example:
        nmf, tr_nmf1, ts_nmf1 = get_decomposition(train.fillna(-100).abs(), test.fillna(-100).abs(), n_component=20, which_method='nmf')
        pca, tr_pca1, ts_pca1 = get_decomposition(train.fillna(-100), test.fillna(-100), n_component=10, which_method='pca')
        svd, tr_svd1, ts_svd1 = get_decomposition(train.fillna(-100), test.fillna(-100), n_component=20, which_method='svd')
    """
    if col_name is None:
        col_name = train.columns

    if test is None:
        complete_df = train[col_name]
    else:
        complete_df = pd.concat([train[col_name], test[col_name]], axis=0)

    if which_method is 'svd':
        method = TruncatedSVD(n_components=n_component, random_state=1234)
    elif which_method is 'nmf':
        method = NMF(n_components=n_component, random_state=1234, alpha=alpha, l1_ratio=l1_ratio)
    elif which_method is 'pca':
        method = PCA(n_components=n_component, random_state=1234, whiten=False)
        try:
            plt.figure()
            plt.plot(np.cumsum(method.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Variance (%)') #for each component
            plt.title('Pulsar Dataset Explained Variance')
            plt.show()
        except:
            pass
    else:
        raise Exception("Please make sure which_method is one of [svd, nmf, [pca]]")

    complete_method = method.fit_transform(complete_df)

    complete_method = pd.DataFrame(data=complete_method)
    complete_method.columns = [which_method+'_'+str(i) for i in range(n_component)]

    train_method = complete_method.iloc[:train.shape[0]]
    if test is None:
        return train_method
    else:
        test_method = complete_method.iloc[train.shape[0]:].reset_index(drop=True)
        return method, train_method, test_method
        














 from sklearn.decomposition import PCA

 def get_pca(n_components=1, whiten=False, random_state=None):
    """
    whiten : bool, optional (default False) When True (False by default) the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
    """
    pass


from sklearn.decomposition import TruncatedSVD
def get_svd_cat_wise(train, test=None, n_components=1, col_name=None):
    """return svd transformation
    Args:
        train, test: dataframe
        col_name: list of column name to be transform [if None, used all column]
        n_component: no of component to be used
    example:
        train_test3 = get_svd_cat_wise(train_test2, n_components=1)
    """
    if col_name is None:
        col_name = train.columns

    if test is None:
        complete_df = train[col_name]
    else:
        complete_df = pd.concat([train[col_name], test[col_name]], axis=0)
    
    complete_svd = pd.DataFrame()
    for col in col_name:
        svd = TruncatedSVD(n_components=n_components)
        
        # if(len(np.unique(complete_df[col])) > 200):
        #     print("please take care of ",col, ". It will raise memory error!")
        tp__ = pd.get_dummies(complete_df[col])
#         print( "==", tp__.shape)
        tp = svd.fit_transform(tp__)
        tp = pd.DataFrame(data=tp)
        tp.columns = [col+'_'+str(i) for i in range(n_components)]
        complete_svd = pd.concat([complete_svd, tp], axis=1)
        print("|", end="")
#     complete_svd = pd.DataFrame(data=complete_svd)
#     complete_svd.columns = [col_name[j]+'_'+str(i)+'_'+str(j) for i in range(n_components) for j in range(len(col_name))]

    if test is None:
        train_svd = complete_svd
    else:
        train_svd = complete_svd.iloc[:train.shape[0]]
        test_svd = complete_svd.iloc[train.shape[0]:].reset_index(drop=True)

    del complete_svd, complete_df
    gc.collect()
    
    if test is None:
        return train_svd
    else:
        return train_svd, test_svd


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