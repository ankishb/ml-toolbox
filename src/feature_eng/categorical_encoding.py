
from category_encoders import BinaryEncoder
from category_encoders import OneHotEncoder
from category_encoders import HashingEncoder
from category_encoders import OrdinalEncoder
from category_encoders import TargetEncoder
from category_encoders.basen import BaseNEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.polynomial import PolynomialEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.backward_difference import BackwardDifferenceEncoder

import numpy as np
import pandas as pd 
import gc

def categorical_enc_type1(train, test, enc_type, col_list, hash_comp=None):
	"""
	enc_type: name of encoding 
		[binary, one_hot, hashing, ordinal, polynomial, backward_diff, sum_enc, base_n]
	col_list: list of column

	#Base-N encoder encodes the categories into arrays of their base-N representation. A base of 1 is equivalent to one-hot encoding (not really base-1, but useful), a base of 2 is equivalent to binary encoding. N=number of actual categories is equivalent to vanilla ordinal encoding.

	example:
		x1, x2 = categorical_enc_type1(train, test, 'hashing', col_list, hash_comp=2)
	"""
	if not isinstance(col_list, list):
		col_list = [col_list]

	if enc_type == 'binary':
		enc = BinaryEncoder(cols=col_list)
	if enc_type == 'one_hot':
		enc = OneHotEncoder(cols=col_list)
	if enc_type == 'hashing':
		enc = HashingEncoder(cols=col_list, n_components=hash_comp, drop_invariant=True)
	if enc_type == 'ordinal':
		enc = OrdinalEncoder(cols=col_list, drop_invariant=True)
	if enc_type == 'polynomial':
		enc = PolynomialEncoder( cols=col_list, drop_invariant=True)
	if enc_type == 'backward_diff':
		enc = BackwardDifferenceEncoder( cols=col_list, drop_invariant=True)
	if enc_type == 'sum_enc':
		enc = SumEncoder(cols=col_list, drop_invariant=True)
	if enc_type == 'base_n':
		enc = BaseNEncoder(cols=col_list, drop_invariant=True, base=1)

	complete_df = pd.concat([train, test], axis=0).reset_index(drop=True)
	enc.fit(complete_df)
	complete_enc = enc.transform(complete_df)

	train_enc = complete_enc.iloc[:train.shape[0]]
	test_enc  = complete_enc.iloc[train.shape[0]:]
	test_enc  = test_enc.reset_index(drop=True)

	del complete_enc, complete_df
	gc.collect()
	return train_enc, test_enc




def categorical_enc_type2(train, test, valid, target_, enc_type, col_list, target_min_leaf=None, target_smoothing=None, loo_sigma=1):
	"""
	enc_type: name of encoding 
		[target_enc, leave_one_out]
	col_list: list of column

	example:
		x1, x2, x3 = categorical_enc_type2(train, test, valid, target_, 'target_enc', col_list, target_min_leaf=3, target_smoothing=2, loo_sigma=1)
		x1.head().append(x2.tail()).append(x3.tail())
	"""
	if not isinstance(col_list, list):
		col_list = [col_list]

	if enc_type == 'target_enc':
		enc = TargetEncoder(cols=col_list, min_samples_leaf=target_min_leaf, smoothing=target_smoothing)
	if enc_type == 'leave_one_out':
		enc = LeaveOneOutEncoder(cols=col_list,random_state=1234, randomized=True, sigma=loo_sigma)
	
	enc.fit(train, target_)
	train_enc = enc.transform(train)
	valid_enc = enc.transform(valid)
	test_enc  = enc.transform(test)

	return train_enc, valid_enc, test_enc






def expanding_mean(train_df, test_df, col_list, target_col, alpha=0, random_state=1234):
	"""
	example:
		x1, x2 = expanding_mean(train_, test, ['p_len', 's_len'], 'target', alpha=0, random_state=1234)

	"""
    train_enc = pd.DataFrame()
    test_enc  = pd.DataFrame()

    global_mean = train_df[target_col].mean()
    for col in col_list:
        # Getting means for test data
        nrows = train_df.groupby(col)[target_col].count()
        target_means = train_df.groupby(col)[target_col].mean()
        target_means_reg = (target_means*nrows + global_mean*alpha)/(nrows+alpha)
        
        # Mapping means to test data
        encoded_test = test_df[col].map(target_means_reg)
        
        # Getting a train encodings
        train_df_shuffled = train_df.sample(frac=1, random_state=random_state)
        cumsum = train_df_shuffled.groupby(col)[target_col].cumsum() - train_df_shuffled[target_col]
        cumcnt = train_df_shuffled.groupby(col).cumcount()
        encoded_train = cumsum/(cumcnt)
        encoded_train.fillna(global_mean, inplace=True)

        train_enc = pd.concat([train_enc, encoded_train], axis=1)
        test_enc = pd.concat([test_enc, encoded_test], axis=1)
    
    return train_enc, test_enc






def kfold_mean_encode(train_df, test_df, col_list, target_col, alpha=0, add_random=False, rmean=0, rstd=0.1, folds=3, random_state=1234):
	"""
	example:
		x1, x2 = kfold_mean_encode(train_, test, ['s_len', 'p_len'], 'target', alpha=0, add_random=False, rmean=0, rstd=0.1, folds=3, random_state=1234)
	"""

    train_enc = pd.DataFrame()
    test_enc  = pd.DataFrame()

    global_mean = train_df[target_col].mean()
    for col in col_list:
        # Getting means for test data
        nrows = train_df.groupby(col)[target_col].count()
        target_means = train_df.groupby(col)[target_col].mean()
        target_means_reg = (target_means*nrows + global_mean*alpha)/(nrows+alpha)

        encoded_test = test_df[col].map(target_means_reg)
        

        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

        valid_fold_enc = np.zeros((train_df.shape[0]))
        for tr_in, val_ind in kfold.split(train_df, train_df[target_col].values):
            X_train, X_valid = train_df.iloc[tr_in], train_df.iloc[val_ind]

            # getting means on data for estimation (all folds except estimated)
            nrows_ = X_train.groupby(col)[target_col].count()
            target_means_ = X_train.groupby(col)[target_col].mean()
            target_means_reg_ = (target_means_*nrows_ + global_mean*alpha)/(nrows_+alpha)
            # Mapping means to estimated fold
            encoded_col_train_part = X_valid[col].map(target_means_reg_)
            if add_random:
                encoded_col_train_part = encoded_col_train_part + normal(loc=rmean, scale=rstd, 
                                                                         size=(encoded_col_train_part.shape[0]))

            valid_fold_enc[val_ind] = encoded_col_train_part.values
        
        valid_fold_enc = pd.DataFrame(data=valid_fold_enc, columns=[col])
        encoded_train = valid_fold_enc#pd.concat(parts, axis=0)
        encoded_train.fillna(global_mean, inplace=True)

        train_enc = pd.concat([train_enc, encoded_train], axis=1)
        test_enc = pd.concat([test_enc, encoded_test], axis=1)
    
    return train_enc, test_enc


"""
averages = X_data.groupby(s)["target"].agg(["mean", "count"])
smoothing_v = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
averages[f] = X_data["target"].mean() * (1 - smoothing_v) + averages["mean"] * smoothing_v
averages.drop(["mean", "count"], axis=1, inplace=True)

np.random.seed(42)
noise = np.random.randn(len(averages[f])) * noise_level
averages[f] = averages[f] + noise

X_train = pd.merge(X_train, averages, how='left', left_on=s, right_index=True)
X_valid = pd.merge(X_valid, averages, how='left', left_on=s, right_index=True)
X_test = pd.merge(X_test, averages, how='left', left_on=s, right_index=True)                       


"""



def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None, val_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series, noise_level)
    



    global_mean = train_df[target_col].mean()
    for col in col_list:
        # Getting means for test data
        nrows = train_df.groupby(col)[target_col].count()
        target_means = train_df.groupby(col)[target_col].mean()
        target_means_reg = (target_means*nrows + global_mean*alpha)/(nrows+alpha)

        encoded_test = test_df[col].map(target_means_reg)
        

def target_encode(train_df, valid_df, test_df, col_name, target_col, min_samples_leaf=1, smoothing=1, noise_level=0, fillna_flag=True):
	"""
	example:
		target_encode(train_, valid, test, 'p_len', 'target')
	"""
    averages = train_df.groupby(col_name)[target_col].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    prior = train_df[target_col].mean()

    # The bigger the count the less full_avg is taken into account
    averages[target_col] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    train_enc = train_df[col_name].map(averages['target'])
    valid_enc = valid_df[col_name].map(averages['target'])
    test_enc  = test_df[col_name].map(averages['target'])

    train_enc = train_enc * (1 + noise_level * np.random.randn(len(train_enc)))
    valid_enc = valid_enc * (1 + noise_level * np.random.randn(len(valid_enc)))
    test_enc  = test_enc * (1 + noise_level * np.random.randn(len(test_enc)))
    
    if fillna_flag:
	    train_enc.fillna(prior, inplace=True)
	    valid_enc.fillna(prior, inplace=True)
	    test_enc.fillna(prior, inplace=True)
    
    return train_enc, valid_enc, test_enc
    



def mean_encode(train_df, test_df, col_list, target_col, alpha=0, add_random=False, rmean=0, rstd=0.1, folds=3, random_state=1234):
	"""
	example:
		mean_encode(train_, test, ['p_len', 's_len'], 'target')
	"""

    train_enc = pd.DataFrame()
    test_enc  = pd.DataFrame()

    for col in col_list:
        
        train_enc_col = np.zeros((train_df.shape[0], folds))
        test_enc_col  = np.zeros((test_df.shape[0], folds))
        
        
        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        for cur_fold,(tr_ind, val_ind) in enumerate(kfold.split(train_df, train_df[target_col].values)):
            X_train, X_valid = train_df.iloc[tr_ind], train_df.iloc[val_ind]

            tr, vl, ts = target_encode(X_train, X_valid, test_df, col, target_col)
            
            train_enc_col[tr_ind, cur_fold] = tr
            train_enc_col[val_ind, cur_fold] = vl
            test_enc_col[:, cur_fold] = ts
            
        encoded_train = pd.DataFrame(data=train_enc_col.mean(axis=1), columns=[col])
        encoded_test  = pd.DataFrame(data=test_enc_col.mean(axis=1), columns=[col])
        

        train_enc = pd.concat([train_enc, encoded_train], axis=1)
        test_enc = pd.concat([test_enc, encoded_test], axis=1)
    
    return train_enc, test_enc










1. cv loop
2. smoothing
3. Random Noise
4. Sorting and calculating expanding mean


1. Regularization CV loop:

y_tr = df['target'].values
skf = StratifiedKFold(y_tr, 5, shuffle=True, random_state=1234)

for tr_idx, val_idx in skf:
	X_tr, X_val = df.iloc[tr_idx], df.iloc[val_idx]
	for col in cols:
		means = X_val[col].map(X_tr.groupby(col).target.mean())
		X_val[col+'mean_target'] = means
	train_new.iloc[val_idx] = X_val

prior = df['target'].mean()
train_new.fillna(prior, inplace=True)

2. smoothing: (mean(target)*nrows + globalmean*alpha)/(nrows+alpha)
alpha: Regularization term
As alpha ~ inf, this formula return global mean 

3. Noise:
	Not a good, because this is unstable method

4. Expanding mean
	- least amount of leakage
	- no hyperparameter
	- irregular encoding quality
	- built in catboost

cumsum = df.groupby(col)['target'].cumsum() - df['target']
cumcnt = df.groupby(col).cumcount()
train_new[col + 'mean_target'] = cumsum/cumcnt

Note: the encoded feature is not uniform, to mak eit uniform, we can
	  use different permutation of data to compute uniform distribution.


def calc_smooth_mean(df, by, on, alpha):
    # Compute the global mean
    globalmean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + alpha * globalmean) / (counts + alpha)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)

# Let’s see what this does in the previous example with a weight of, say, 10.

df['x_0'] = calc_smooth_mean(df, by='x_0', on='y', alpha=10)
df['x_1'] = calc_smooth_mean(df, by='x_1', on='y', alpha=10)


# Reference: https://www.coursera.org/learn/competitive-data-science/lecture/D09Jb/extensions-and-generalizations
For regression task:
	use percentile, std, distribution-bins
For multi-class_:
	use one vs all classifiers (output will be n feature, one for each class_)
For many-to-many relational data:
	use long representation of data and calculate vector of encoding.

	user_id, app_checked, target
	user1, [app1, app2, app7], 1
	user2, [app2, app4], 0

	Long representation:
	user1, app1, 1
	user1, app2, 1
	user1, app7, 1
	user2, app2, 0
	user2, app4, 0

	use app_checked and target feature and prepare mean_target feature, 
	and collect them in a vector form on basis of user_id.

For time-series:
	use target statistics to prepare more complicated features.
	statistics such as rolling 	


Interaction Based feature:
	Follow each tree, and check how many times two features have interact, 
	and that features can be used for mean encoding.
	Note: Catboost do such interaction based feature by itself.

	feature1 -----> feature2
		|
		------> feature3


# import category_encoders as ce
# import pandas as pd
# from sklearn.datasets import load_boston, load_iris

# # prepare some data
# bunch = load_iris()
# y = bunch.target
# X = pd.DataFrame(bunch.data, columns=['s_length', 's_width', 'p_length', 'p_width'])

# X.shape, y.shape
# import numpy as np
# idx = np.random.permutation(y.shape[0])
# X = X.iloc[idx].reset_index(drop=True)
# y = y[idx]


# X2 = X.copy()
# y2 = y.copy()
# X = X2.iloc[:100]
# y = y2[:100]
# X1 = X2.iloc[-50:]
# y1 = y2[-50:]
# X.shape, y.shape, X1.shape, y.shape

# tr = X.copy()
# ts = X1.copy().reset_index(drop=True)
# complete = pd.concat([tr, ts], axis=0).reset_index(drop=True)
# tr.shape, ts.shape, complete.shape


# min_samples_leaf=1, smoothing=1)



