    
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
    