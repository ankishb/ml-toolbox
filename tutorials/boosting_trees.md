
## XGBOOST:
Boosted Tree Iteratively fit a new model in the current representation or predicted output as
```
y^0 = 0
y^1 = y^0 + f1(x)
....
y^t = y^(t-1) + ft(x)
```

> ft(x) = (y^t - y^(t-1)) = error

Objective function = loss(y, y^t) + reg_loss

obj(t) = loss(y,y^(t)) + Ω(fi) 
    = loss(y, y^(t−1) + ft(x))^2 + Ω(ft)

If we consider using mean squared error (MSE) as our loss function, the objective becomes
obj(t) = (y −(y^(t−1) + ft(x)))^2 + Ω(f) 
= [2(y^(t−1)−y) ft(x) + ft(x)^2] + Ω(ft) 

In the general case, we take the Taylor expansion of the loss function up to the second order:
obj(t) = [loss(y,y^(t−1)) + grad ft(x) + hessian f^2t(x)] + Ω(ft)

grad = ∂y^(t−1)loss(y,y^(t−1))
hessian = ∂2y^(t−1)loss(y,y^(t−1))


For reg_loss, it use complexity of tree(number of leafs and weights at leaf nodes)
reg_loss = γT+ w^2

Overall Objective function: [grad*weight + (hessian + γ)*weight^2] + γ*T

Optimal weight: -grad/(hessian + γ)
So obejctive: grad^2/(hessian + γ) + γ * T

Where first terms tell how good it the split is or ho good the tree structure is

To split at node:
gain  = [gain of left child + gain of right child - gain, if we don't split] - complexity at that split

where gain: grad^2/(hessian + γ)

- xgboost doesn't handle categorical variable
- use dummy variable/one hot encoding for cat variable

### Unique feature of xgboost
1. regularization
2. handling sparse data
3. cache awareness, to save compuitation from recomputing gradient
4. parallel learning
5. scalable (used by CERN on petabytes of data)
6. weighted quantile sketch (weight column to each sample row)
7. out of core usuage(optimize the disk space for huge dataset)

---


XGBoost: It need all data in the int/float form
LightGBM: It can handle categorical columns, if dtype is chosen to be category, or if we explicitly provide columns name, shown in following example
Note: LightGBM can not work on string dtypes

```python
lg = lgb.LGBMClassifier(silent=False, categorical_feature=cat_cols)
```
or better
```python
cat_col = train.select_dtypes('object').columns.tolist()

d_train = lgb.Dataset(
    X_train, 
    label = y_train, 
    feature_name = list(X_train.columns), 
    categorical_feature = cat_cols
)
```

To handle missing values, use `use_missing=True`
Use min_data_per_group, cat_smooth to deal with over-fitting (when #data is small or #category is large).

`min_data_per_group` (default = 100)

    minimal number of data per categorical group


`cat_l2` (default = 10.0)

    L2 regularization in categorcial split

`cat_smooth` (default = 10.0)

    this can reduce the effect of noises in categorical features, especially for categories with few data






Catboost:

`grow_policy`:

    `SymmetricTree` —A tree is built level by level until the specified depth is reached. On each iteration, all leaves from the last tree level are split with the same condition. The resulting tree structure is always symmetric.
    `Depthwise` — A tree is built level by level until the specified depth is reached. On each iteration, all non-terminal leaves from the last tree level are split. Each leaf is split by condition with the best loss improvement.
    `Lossguide` — A tree is built leaf by leaf until the specified maximum number of leaves is reached. On each iteration, non-terminal leaf with the best loss improvement is split.


There are some category specific parameters, such as cat_encoding.


Can be used only with the Lossguide and Depthwise growing policies.



## Parameter Tuning

For heavily unbalanced datasets such as 1:10000:

    max_bin: keep it only for memory pressure, not to tune (otherwise overfitting)
    learning rate: keep it only for training speed, not to tune (otherwise overfitting)
    n_estimators: must be infinite (like 9999999) and use early stopping to auto-tune (otherwise overfitting)
    num_leaves: [7, 4095]
    max_depth: [2, 63] and infinite (I personally saw metric performance increases with such 63 depth with small number of leaves on sparse unbalanced datasets)
    scale_pos_weight: [1, 10000] (if over 10000, something might be wrong because I never saw it that good after 5000)
    min_child_weight: [0.01, (sample size / 1000)] if you are using logloss (think about the hessian possible value range before putting "sample size / 1000", it is dataset-dependent and loss-dependent)
    subsample: [0.4, 1]
    bagging_freq: only 1, keep as is (otherwise overfitting)
    colsample_bytree: [0.4, 1]
    is_unbalance: false (make your own weighting with scale_pos_weight)
    USE A CUSTOM METRIC (to reflect reality without weighting, otherwise you have weights inside your metric with premade metrics like xgboost)

Never tune these parameters unless you have an explicit requirement to tune them:

    Learning rate (lower means longer to train but more accurate, higher means smaller to train but less accurate)
    Number of boosting iterations (automatically tuned with early stopping and learning rate)
    Maximum number of bins (RAM dependent)


## LightGBM docs summary:

### For Faster Speed:
1. Use bagging by setting `bagging_fraction` and `bagging_freq`
2. Use feature sub-sampling by setting `feature_fraction`
3. Use small `max_bin`
4. Use `save_binary` to speed up data loading in future learning

### For Better Accuracy:
1. Use large `max_bin` (may be slower)
2. Use small `learning_rate` with large `num_iterations`
3. Use large `num_leaves` (may cause over-fitting)
4. Use bigger training data
5. Try dart

### Deal with Over-fitting:
1. Use small `max_bin`
2. Use small `num_leaves`
3. Use `min_data_in_leaf` and `min_sum_hessian_in_leaf`
4. Use bagging by set `bagging_fraction` and `bagging_freq`
5. Use feature sub-sampling by set `feature_fraction`
6. Use bigger training data
7. Try `lambda_l1`, `lambda_l2` and `min_gain_to_split` for regularization
8. Try `max_depth` to avoid growing deep tree

Reference: https://sites.google.com/view/lauraepp/parameters












