

##############################################################################################################
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os, gc
# os.listdir()

##############################################################################################################



##############################################################################################################
complete_df = pd.concat([train_df, test_df], axis=0)
col_to_use = complete_df.columns

complete_df.apply(lambda x: pd.Series.value_counts(x).shape[0])
##############################################################################################################




##############################################################################################################
# Function which returns subset or r length from n 
from itertools import combinations 

def rSubset(arr, r): return list(combinations(arr, r)) 

count_all = 0
for r in range(2,4):
    print(r, " count: ==> ", end=" ")
    count = 0
    for inter in list(combinations(col_to_use, r)):
        new_col_name = "+".join(inter)
        count_all += 1
        count += 1
    print(count)
print("total_combinations: ", count)
##############################################################################################################




##############################################################################################################
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_df, target, 
                                                     random_state=1234, 
                                                     stratify=target, 
                                                     test_size=0.25)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
##############################################################################################################






##############################################################################################################
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def run_rf(X_train, X_valid, y_train, y_valid):

    clf = RandomForestClassifier(
                            n_estimators = 20, 
                            criterion = 'gini', 
#                             max_depth = depth, 
                            # min_samples_split = 2, 
                            min_samples_leaf = 3, 
                            # min_weight_fraction_leaf = 0.0, 
                            max_features = 0.5, 
                            # max_leaf_nodes = None, 
                            # min_impurity_decrease = 0.0, 
#                                 bootstrap = True, 
                            oob_score = True, 
                            n_jobs = 4, 
                            random_state = 1337, 
                            verbose = 1, 
                            class_weight = 'balanced')#3.607)
    clf.fit(X_train, y_train)
        
    valid_pred = clf.predict_proba(X_valid)[:,1]
    score = roc_auc_score(y_valid, valid_pred)
    print("CV score: {:<8.5f}".format(score))

    feat_name = X_train.columns
    feat_imp  = clf.feature_importances_

    rf_imp = pd.DataFrame(data=[list(feat_name), list(feat_imp)]).T
    rf_imp.columns=['feature', 'importance']
    rf_imp = rf_imp.sort_values(by='importance', ascending=False).head(20)

    return score, clf, rf_imp
##############################################################################################################





##############################################################################################################
from sklearn.preprocessing import StandardScaler, MinMaxScaler

stdc = StandardScaler()
min_max = MinMaxScaler()
# for col in cols_to_change:
#     complete_df[col] = stdc.fit_transform(complete_df[col])
    
X1 = stdc.fit_transform(complete_df[cols_to_change])
X2 = min_max.fit_transform(complete_df[cat_cols])

X1 = pd.DataFrame(data=X1, columns=cols_to_change)
X2 = pd.DataFrame(data=X2, columns=cat_cols)

##############################################################################################################
import re
def split_it(x):
    return re.findall('[:\+\-\*\/]', x)# 

# save_feat = pd.DataFrame()
# for ii,imp in enumerate(rf_imps):
#     imp['position'] = np.arange(imp.shape[0])+1
#     imp['position'] = imp.apply(lambda x: x[2] if split_it(x[0]) else 0, axis=1)
#     imp['which_file'] = ii
#     save_feat = pd.concat([save_feat, imp[imp.position != 0]], axis=0)
    
#     if ii%10 == 0:
#         print("reach here at ",ii)

##############################################################################################################
rf_scores = []
rf_imps = []
rf_clf = []
count = 0
for idx,inter in enumerate(list(combinations(col_to_use, 2))):
    if idx%40 == 0:
        if idx != 0:
            print(idx, "==", end=" ")
            tr = pd.concat([X_train.reset_index(drop=True), 
                feature_df.iloc[:X_train.shape[0],:].reset_index(drop=True)], axis=1)
            ts_feat = feature_df.iloc[X_train.shape[0]:X_train.shape[0]+X_valid.shape[0],:]
            ts = pd.concat([X_valid.reset_index(drop=True), 
                            ts_feat.reset_index(drop = True)], axis=1)
            print(tr.shape, ts.shape)
            # reduce_mem_usage_wo_print(tr)
            # reduce_mem_usage_wo_print(ts)
            
            score, clf, imp = run_rf(tr.fillna(0), ts.fillna(0), y_train, y_valid)

            imp['position'] = np.arange(imp.shape[0])+1
            imp['position'] = imp.apply(lambda x: x[2] if split_it(x[0]) else 0, axis=1)
            feature_df = feature_df[imp[imp.position != 0].feature.values]

            rf_scores.append(score)
            rf_clf.append(clf)
            rf_imps.append(imp)
            
            file_name = 'store_data/new_feat/int_inter_2way/num_inter_2way_'+str(idx)+'.csv'
            feature_df.to_csv(file_name, index=None)
            del tr, ts_feat, ts
            gc.collect()
            print(file_name)
            
        feature_df = pd.DataFrame()
        gc.collect()
        print(idx)
    new_col_name1 = "+".join(inter)
    new_col_name2 = "-".join(inter)
    new_col_name3 = "*".join(inter)
    new_col_name4 = "/".join(inter)+'_pre'
    new_col_name5 = "/".join(inter)+'_post'
    
    feature_df[new_col_name1] = complete_df[inter[0]] + complete_df[inter[1]]
    feature_df[new_col_name2] = complete_df[inter[0]] - complete_df[inter[1]]
    feature_df[new_col_name3] = complete_df[inter[0]] * complete_df[inter[1]]
#     feature_df[new_col_name3] = stdc.fit_transform(complete_df[inter[0]])*stdc.fit_transform(complete_df[inter[1]])
    try:
        feature_df[new_col_name4] = complete_df[inter[0]] / (1+complete_df[inter[1]])
    except:
        try:
            feature_df[new_col_name4] = complete_df[inter[0]] / (2+complete_df[inter[1]])
        except:
            print("could not do it")
            pass
        
    feature_df[new_col_name5] = complete_df[inter[1]] / (1+complete_df[inter[0]])
    
    if (feature_df.shape[1]+1)%90 == 0:
        reduce_mem_usage_wo_print(feature_df)
        gc.collect()
    count += 1
file_name = 'store_data/new_feat/int_inter_2way/num_inter_2way_'+str(idx)+'.csv'
feature_df.to_csv(file_name, index=None)
print(idx, "==", end=" ")
print(file_name)
print(count)

gc.collect()
##############################################################################################################





##############################################################################################################
import pandas as pd
import numpy as snp
import glob

int_inter_2way = glob.glob('store_data/new_feat/int_inter_2way/*')
int_inter_3way = glob.glob('store_data/new_feat/int_inter_3way/*')
len(int_inter_2way), len(int_inter_3way)





total_feat = 0
feature_df = pd.DataFrame()
idx = 0
for file_no,file in enumerate(int_inter_3way):
    if feature_df.shape[1] > 100:
        print(feature_df.shape, "==>", end=" ")
        file_name = 'store_data/new_feat/int_inter_3way_new/cat_inter_3way_'+str(idx)+'.csv'
        feature_df.iloc[:,:100].to_csv(file_name, index=None)
        feature_df = feature_df.iloc[:,100:]
        print(feature_df.shape)
        idx += 1
        
    try:
        feat = pd.read_csv(file)
    except:
        print("===",file,"===empty file")
        pass
#     ####### Label Encoding ########
#     le = LabelEncoder()
#     feat.fillna(0, inplace=True)
#     for col in feat.columns:
#         try:
#             feat[col] = le.fit_transform(feat[col])
#         except:
#             print(file, "==", col)
#     ####### Label Encoding ########
    
    feature_df = pd.concat([feature_df, feat], axis=1)
    print(file_no, "==", file)
    
print(feature_df.shape, "==>", end=" ")
file_name = 'store_data/new_feat/int_inter_3way_new/cat_inter_3way_'+str(idx)+'.csv'
feature_df.to_csv(file_name, index=None)
print(feature_df.shape)
print("===========finish===========")
##############################################################################################################


