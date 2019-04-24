import pandas as pd
import numpy as np
import gc



def fill_outlier(data, flag, filling=None, cols=None):
    """fill outlier either with nan or bounds values
    Args:
        df  : feature
        filling: ['nan','bound'] (by default, it is 'bound')
        flag: string ['lower','upper','both']
    return: dataframe
    example:
        fill_outlier(X1, "both", filling="np.nan")
    """
    df = data.copy()
    df1 = df.dropna()
    if cols is None:
        cols = list(df.columns)
        # select appropriate columns by dtypes
    cols = df.select_dtypes(exclude='object').columns
    for col in cols:
        lower_bound = np.percentile(df1[col], q=1)
        upper_bound = np.percentile(df1[col], q=99)
        print(col, "(", df[col].dtype, ") ==>","low: ", np.round(lower_bound,2), 
              " high: ", np.round(upper_bound,2))
        
        if filling is None or filling == "bound":
            lower_bound_fill = lower_bound
            upper_bound_fill = upper_bound
        else:
            lower_bound_fill = np.nan
            upper_bound_fill = np.nan
        if flag == 'upper':
            df[col] = np.where(df[col]>upper_bound, upper_bound_fill, df[col])
        elif flag == 'lower':
            df[col] = np.where(df[col]<lower_bound, lower_bound_fill, df[col])
        else: 
            # when both are selected
            df[col] = np.where(df[col]>upper_bound, upper_bound_fill, df[col])
            df[col] = np.where(df[col]<lower_bound, lower_bound_fill, df[col])
    print("null count: ", df.isnull().sum().values)
#     new = pd.DataFrame(data=new, columns=df1.columns)
    return df

# new = remove_and_fill_outlier(train_test.disbursed_amount, flag='upper')
# new.shape
# np.clip(train_test.disbursed_amount, lower_bound, upper_bound).shape, train_test.shape

def get_quantile(df, col, q1, q2):
    """compute quantile range
    args:
        col: col name
        q1: lower quantile percentile
        q2: upper quantile percentile
    """
    df1 = df[[col]].dropna()
    lower_bound = np.percentile(df1, q=q1)
    upper_bound = np.percentile(df1, q=q2)
    print("low: ", np.round(lower_bound,3), "  ", end=" ")
    print("high: ", np.round(upper_bound,3))


def remove_outlier(data, flag, cols=None):
    """fill outlier either with nan or bounds values
    Args:
        df  : dataframe
        flag: string ['lower','upper','both']
    return: dataframe
    example:
        remove_outlier(X, flag="upper")
    """
    df = data.copy()
    df1 = df.dropna()
    if cols is None:
        cols = list(df.columns)
        # select appropriate columns by dtypes
    cols = df.select_dtypes(exclude='object').columns
    print("Initial shape: ", df.shape) 
    for col in cols:
        lower_bound = np.percentile(df1[col], q=1)
        upper_bound = np.percentile(df1[col], q=99)
        print(col, "(", df[col].dtype, ") ==>","low: ", np.round(lower_bound,2), 
              " high: ", np.round(upper_bound,2), " outliers: ", end=" ")
        
        init = df.shape[0]
        if flag == 'upper':
            df = df[df[col]<upper_bound]
        elif flag == 'lower':
            df = df[df[col]>lower_bound]
        else: # when both are selected
            df = df[df[col]<upper_bound]
            df = df[df[col]>lower_bound]
        final = df.shape[0]
        print(init-final)
    print("final shape: ", df.shape)
    return df

# train_test = remove_outlier(train_test, 'disbursed_amount','upper')
# train_test = remove_outlier(train_test, 'asset_cost','upper')