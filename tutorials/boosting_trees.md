

XGBoost: It need all data in the int/float form
LightGBM: It can handle categorical columns, if dtype is chosen to be category, or if we explicitly provide columns name, shown in following example
Note: LightGBM can not work on string dtypes

```python
lg = lgb.LGBMClassifier(silent=False, categorical_feature=cat_cols)
```
or better
```python
cat_col = train.select_dtypes('object').columns.tolist()

d_train = lgb.Dataset(X_train, label = y_train, 
                      feature_name = list(X_train.columns), 
                      categorical_feature = cat_cols)
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