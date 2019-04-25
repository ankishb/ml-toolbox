
import numpy as np 
import pandas as pd 
import gc
import category_encoders as ce

encoder = ce.BackwardDifferenceEncoder(cols=[...])
encoder = ce.BinaryEncoder(cols=[...])
encoder = ce.HashingEncoder(cols=[...], n_components=8)
encoder = ce.HelmertEncoder(cols=[...])
encoder = ce.OneHotEncoder(cols=[...])
encoder = ce.OrdinalEncoder(cols=[...])
encoder = ce.SumEncoder(cols=[...])
encoder = ce.PolynomialEncoder(cols=[...])
encoder = ce.BaseNEncoder(cols=[...])
.basen.BaseNEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True, base=2, impute_missing=True, handle_unknown='impute')[source]

    #Base-N encoder encodes the categories into arrays of their base-N representation. A base of 1 is equivalent to one-hot encoding (not really base-1, but useful), a base of 2 is equivalent to binary encoding. N=number of actual categories is equivalent to vanilla ordinal encoding.

encoder = ce.TargetEncoder(cols=[...])
encoder = ce.LeaveOneOutEncoder(cols=[...])

encoder.fit(X, y)
X_cleaned = encoder.transform(X_dirty)



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
enc = ce.TargetEncoder(cols=['s_width', 'p_width'], 
                       min_samples_leaf=3, smoothing=10).fit(tr, y)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))


enc = ce.leave_one_out.LeaveOneOutEncoder(cols=['s_width', 'p_width']).fit(tr, y)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))


enc = ce.leave_one_out.LeaveOneOutEncoder(cols=['s_width', 'p_width'],
                                          random_state=1234, 
                                          randomized=True, 
                                          sigma=1).fit(tr, y)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))



enc = ce.BinaryEncoder(cols=['CHAS', 'RAD']).fit(complete)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))




enc = ce.OneHotEncoder(cols=['CHAS', 'RAD']).fit(complete)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))





enc = ce.HashingEncoder(cols=['CHAS', 'RAD'], 
                        n_components=6,
                        drop_invariant=True).fit(complete)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))




enc = ce.OrdinalEncoder(cols=['CHAS', 'RAD'],
                        drop_invariant=True).fit(complete)
print(enc.category_mapping)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))





enc = ce.polynomial.PolynomialEncoder(
    cols=['CHAS', 'RAD'], drop_invariant=True).fit(complete)
print(enc.mapping)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))





enc = ce.backward_difference.BackwardDifferenceEncoder(
    cols=['s_width'], drop_invariant=True).fit(complete)
print(enc.get_params)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))




enc = ce.sum_coding.SumEncoder(cols=['CHAS', 'RAD'], drop_invariant=True).fit(complete)
print(enc.mapping)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))




enc = ce.basen.BaseNEncoder(cols=['CHAS', 'RAD'], 
                            drop_invariant=True,
                            base=1).fit(complete)
enc.transform(tr).head(5).append(enc.transform(ts).head(5))
