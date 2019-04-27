    
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def get_normalization(train_df, test_df, which_norm=None, int_cols=None, cat_cols=None):
    """ normalization of data features
    Args:
        train_df, test_df
        which_norm: normalization for numerical columns, ['stdc', 'min-max']
        int_cols: list of numerical_columns
        cat_cols: list of categorical columns
    Return:
        standarization scaled version for numeric data
        min-max scaled version for categorical data
    example:

        from sklearn import datasets
        iris = datasets.load_iris()
        X = iris.data; y = iris.target
        X = pd.DataFrame(data=X, columns=iris['feature_names'])
        get_normalization(X.iloc[:10,:], X.iloc[10:15,:], 'min-max',int_cols=X.columns)
        get_normalization(X.iloc[:10,:], X.iloc[10:15,:], int_cols='sepal width (cm)')
        get_normalization(X.iloc[:10,:], X.iloc[10:15,:], int_cols=X.columns)
    """
    complete_df = pd.concat([train_df, test_df], axis=0)

    stdc = StandardScaler()
    min_max = MinMaxScaler()

    if int_cols is not None:
        if which_norm == 'stdc': norm = StandardScaler()
        elif which_norm == 'min-max': norm = MinMaxScaler()
        else: 
            norm = StandardScaler()
            print("by default: stdc norm is running")
        try: 
            complete_df[int_cols] = norm.fit_transform(complete_df[int_cols])
        except: 
            # expecting a 2d array, but passed i-d array
            complete_df[[int_cols]] = norm.fit_transform(complete_df[[int_cols]])
        print("done with stdc normalization on numerical columns")
        
    if cat_cols is not None:
        norm = MinMaxScaler()
        try: 
            complete_df[cat_cols] = norm.fit_transform(complete_df[cat_cols])
        except:
            complete_df[[cat_cols]] = norm.fit_transform(complete_df[[cat_cols]])
        print("done with min-max normalization on categorical columns")

    train_df_new = complete_df.iloc[:train_df.shape[0],:]
    test_df_new  = complete_df.iloc[train_df.shape[0]:,:]
#     print(train_df_new.shape, test_df_new.shape)
    
    test_df_new = test_df_new.reset_index(drop=True)
    
    del complete_df
    gc.collect()
    return train_df_new, test_df_new
    


from sklearn.preprocessing import QuantileTransformer
This method transforms the features to follow a uniform or a normal distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values. It also reduces the impact of (marginal) outliers: this is therefore a robust preprocessing scheme.

n_quantiles : int, optional (default=1000)
    Number of quantiles to be computed. It corresponds to the number of landmarks used to discretize the cumulative distribution function.

output_distribution : ['uniform','normal'], optional (default=’uniform’)
    Marginal distribution for the transformed data

subsample : int, optional (default=1e5)
    Maximum number of samples used to estimate the quantiles for computational efficiency. Note that the subsampling procedure may differ for value-identical sparse and dense matrices.

quant_trans = QuantileTransformer(n_quantiles=1000, 
    output_distribution='uniform', 
    subsample=100000, 
    random_state=1234)

"""
import numpy as np
from sklearn.preprocessing import QuantileTransformer
rng = np.random.RandomState(0)
X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
qt = QuantileTransformer(n_quantiles=10, random_state=0)
qt.fit_transform(X) 
"""









from sklearn.preprocessing import RobustScaler

Scale features using statistics that are robust to outliers.

This Scaler removes the median and scales the data according to the quantile range IQR.

Median and interquartile range are then stored to be used on later data using the transform method.


with_centering : boolean, True by default
    If True, center the data before scaling
with_scaling : boolean, True by default
    If True, scale the data to interquartile range.
quantile_range : tuple (q_min, q_max)

RobustScaler(with_centering=True, 
    with_scaling=True, 
    quantile_range=(25.0, 75.0))

"""
from sklearn.preprocessing import RobustScaler
X = [[ 1., -2.,  2.],
     [ -2.,  1.,  3.],
     [ 4.,  1., -2.]]
transformer = RobustScaler().fit(X)
transformer.transform(X)
"""