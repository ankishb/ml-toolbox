import pandas as pd
import numpy as np
import gc



def fill_outlier(data, l_per=1, h_per=99, flag='both', filling='bound', cols=None):
    """fill outlier either with nan or bounds values
    Args:
        df  : table
        l_per: lower-percentile
        h_per: higher-percentile
        filling: ['nan','bound'] (by default, it is 'bound')
        flag: string ['lower','upper','both'] (by default, it is 'both')
    return: dataframe
    example:
        fill_outlier(X1, "both", filling="np.nan")
    """
    print("-----outlier handling------")
    df = data.copy()
#     df1 = df.dropna()
    collect_cols = []
    if cols is None:
        cols = list(df.columns)
        # select appropriate columns by dtypes
    cols = df.select_dtypes(exclude='object').columns
    for col in cols:
        try:
            lower_bound = np.percentile(df[col].dropna(), q=l_per)
            upper_bound = np.percentile(df[col].dropna(), q=h_per)
#             print("{} ( {} ) ==> low: {} high: {}".format(col, df[col].dtype, np.round(lower_bound,2), np.round(upper_bound,2)))

            if filling is 'bound':
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
        except:
            collect_cols.append(col)
#     print("null count: ", df.isnull().sum().values)
    if(len(collect_cols) > 0):
        print("There are some columns, which needed extra care: \n\n", collect_cols)
    print("\n-----Done with outlier handling-----")
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
    lower_bound = np.round(lower_bound,3)
    upper_bound = np.round(upper_bound, 3)
    min_ = np.round(np.min(df1[col]), 3)
    max_ = np.round(np.max(df1[col]), 3)
    print("{4:<25} min: {0:<10} max: {1:<10} low: {2:<10} high: {3:<10}".format(min_, max_, lower_bound, upper_bound, col))



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