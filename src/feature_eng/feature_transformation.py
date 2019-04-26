

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



