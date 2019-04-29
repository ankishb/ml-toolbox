start_time = time.time()
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
def impact_coding(data, feature, target='y'):
    '''
    In this implementation we get the values and the dictionary as two different steps.
    This is just because initially we were ignoring the dictionary as a result variable.
    
    In this implementation the KFolds use shuffling. If you want reproducibility the cv 
    could be moved to a parameter.
    '''
    n_folds = 10
    n_inner_folds = 10
    impact_coded = pd.Series()
    
    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
#     kf = KFold(n_splits=n_folds, shuffle=True, random_state=2019)
    kf = GroupKFold(n_splits=n_folds)
    oof_mean_cv = pd.DataFrame()
    split = 0
#     for infold, oof in kf.split(data[feature]):
    for infold, oof in kf.split(data[feature], groups=data['RescuerID']):
            impact_coded_cv = pd.Series()
#             kf_inner = KFold(n_splits=n_inner_folds, shuffle=True, random_state=2019)
            kf_inner = GroupKFold(n_splits=n_inner_folds)
            inner_split = 0
            inner_oof_mean_cv = pd.DataFrame()
            oof_default_inner_mean = data.iloc[infold][target].mean()
            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold], groups=data['RescuerID'].iloc[infold]):
                # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()

                # Also populate mapping (this has all group -> mean for all inner CV folds)
                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
                inner_split += 1

            # Also populate mapping
            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
            split += 1
            
            impact_coded = impact_coded.append(data.iloc[oof].apply(
                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
                                      if x[feature] in inner_oof_mean_cv.index
                                      else oof_default_mean
                            , axis=1))

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean  

cat_enc_cols = ['Color1','Color2','Color3','Dewormed','FurLength','Gender',\
                'Health','MaturitySize','Sterilized','Type','Vaccinated', \
                'State', 'VideoAmt', 'Fee','main_breed_Type', 'second_breed_Type']

train_enc = pd.DataFrame()
test_enc = pd.DataFrame()
impact_coding_map = {}
for f in cat_enc_cols:
    print("Impact coding for {}".format(f))
    train_enc["im_enc_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(
                                                                                        X_train_non_null1,
                                                                                        f, target="AdoptionSpeed")
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_mean = impact_coding_map[f]
    test_enc["im_enc_{}".format(f)] = X_test_non_null1.apply(lambda x: \
                                                mapping[x[f]] if x[f] in mapping else default_mean, axis=1)

print("done with Impact encoding in time", (time.time() - start_time)/60)









"""
from sklearn.model_selection import StratifiedKFold, KFold
def impact_coding(data, feature, target='y'):
    '''
    In this implementation we get the values and the dictionary as two different steps.
    This is just because initially we were ignoring the dictionary as a result variable.
    
    In this implementation the KFolds use shuffling. If you want reproducibility the cv 
    could be moved to a parameter.
    '''
    n_folds = 10
    n_inner_folds = 10
    impact_coded = pd.Series()
    
    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2019)
    oof_mean_cv = pd.DataFrame()
    split = 0
    for infold, oof in kf.split(data[feature]):
            impact_coded_cv = pd.Series()
            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True, random_state=2019)
            inner_split = 0
            inner_oof_mean_cv = pd.DataFrame()
            oof_default_inner_mean = data.iloc[infold][target].mean()
            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
                # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
                impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
                            lambda x: oof_mean[x[feature]]
                                      if x[feature] in oof_mean.index
                                      else oof_default_inner_mean
                            , axis=1))

                # Also populate mapping (this has all group -> mean for all inner CV folds)
                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
                inner_split += 1

            # Also populate mapping
            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
            split += 1
            
            impact_coded = impact_coded.append(data.iloc[oof].apply(
                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
                                      if x[feature] in inner_oof_mean_cv.index
                                      else oof_default_mean
                            , axis=1))

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean  
    
    
def frequency_encoding(df, col_name):
    new_name = "{}_cnt".format(col_name)
    new_col_name = "{}_freq".format(col_name)
    grouped = df.groupby(col_name).size().reset_index(name=new_name)
    df = df.merge(grouped, how = "left", on = col_name)
    df[new_col_name] = df[new_name]/df[new_name].count()
    del df[new_name]
    gc.collect()
    return df


cat_enc_cols = ['Color1','Color2','Color3','Dewormed','FurLength','Gender',\
                'Health','MaturitySize','Sterilized','Type','Vaccinated', \
                'State', 'AgeCat', 'VideoAmt', 'Fee','main_breed_Type', 'second_breed_Type']
# categorical_features = ["Type", "Breed1", "Breed2", "Color1" ,"Color2", "Color3", "State"]
# X_train_non_null1[cat_enc_cols] = X_train_non_null1[cat_enc_cols].astype('int')
train_enc = pd.DataFrame()
test_enc = pd.DataFrame()
impact_coding_map = {}
for f in cat_enc_cols:
    print("Impact coding for {}".format(f))
    train_enc["im_enc_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(
                                                                                        X_train_non_null1,
                                                                                        f, target="AdoptionSpeed")
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_mean = impact_coding_map[f]
    test_enc["im_enc_{}".format(f)] = X_test_non_null1.apply(lambda x: \
                                                mapping[x[f]] if x[f] in mapping else default_mean, axis=1)

print("done with Impact encoding")

for cat in cat_enc_cols:
    train_enc1 = frequency_encoding(X_train_non_null1, cat)
    test_enc1 = frequency_encoding(X_test_non_null1, cat)

print("done with freq encoding")
"""