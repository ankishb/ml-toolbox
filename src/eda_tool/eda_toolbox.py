
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline

def boxplot_it(df, col_x, col_y):
    """
    Args:
        df      : data-frame
        col_x   : columnn x for boxplot
        col_y   : columnn y for boxplot
    return: 
        distplot with flag[0/1]
    """
    plt.figure(figsize=(16,4))
    sns.boxplot(y=col_y, x=col_x, data=df)


def distplot_it(df, flag_col, col):
    """
    Args:
        df      : data-frame
        col     : columnn for distplot
        flag_col: flag to represent train or test
    return: 
        distplot with flag[0/1]
    """
    plt.figure(figsize=(16,4))
    sns.distplot(df[df[flag_col] == 1][col], hist=False, label='train')
    sns.distplot(df[df[flag_col] == 0][col], hist=False, label='test')
    _ = plt.xticks(rotation='vertical')
    
distplot_it(train_test, 'train_flag', 'asset_cost')



def countplot_it(df, flag_col, col):
    """
    Args:
        df      : data-frame
        col     : columnn for distplot
        flag_col: flag to represent train or test
    return: 
        distplot with flag[0/1]
    """
	plt.figure(figsize=(16,4))
	sns.countplot(x=col, hue=flag_col, data=train_test)
	_ = plt.xticks(rotation='vertical')

    