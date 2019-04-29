


########################### target_encode Another-kaggle #########################
########################### target_encode Another-kaggle #########################
########################### target_encode Another-kaggle #########################
def perform_single_train(data, hyper):

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_valid = data["X_valid"]
    y_valid = data["y_valid"]
    X_test = data["X_test"]
    
    lgb_pars = hyper["lgb_pars"]
    features = hyper["features"]
    
    rounds = hyper["rounds"]
    early = hyper["early"]

    noise_level = hyper["noise_level"]
    smoothing = hyper["smoothing"]
    min_samples_leaf= hyper["min_samples_leaf"]

    X_data = X_train.copy()
    X_data["target"] = y_train

    X_train_c=X_train.copy()
    X_valid_c=X_valid.copy()
    X_test_c=X_test.copy()

    for f in features:
        s = f.split("_add_")
        if (len(s) == 2):
            c1 = s[0]
            c2 = s[1]
            X_train[f] = X_train_c[c1] + X_train_c[c2]
            X_valid[f] = X_valid_c[c1] + X_valid_c[c2]
            X_test[f] = X_test_c[c1] + X_test_c[c2]

        s = f.split("_sub_")
        if (len(s) == 2):
            c1 = s[0]
            c2 = s[1]
            X_train[f] = X_train_c[c1] - X_train_c[c2]
            X_valid[f] = X_valid_c[c1] - X_valid_c[c2]
            X_test[f] = X_test_c[c1] - X_test_c[c2]

        s = f.split("_mul_")
        if (len(s) == 2):
            c1 = s[0]
            c2 = s[1]
            X_train[f] = X_train_c[c1] * X_train_c[c2]
            X_valid[f] = X_valid_c[c1] * X_valid_c[c2]
            X_test[f] = X_test_c[c1] * X_test_c[c2]

        s = f.split("_div_")
        if (len(s) == 2):
            c1 = s[0]
            c2 = s[1]
            X_train[f] = X_train_c[c1] / X_train_c[c2]
            X_valid[f] = X_valid_c[c1] / X_valid_c[c2]
            X_test[f] = X_test_c[c1] / X_test_c[c2]

        s = f.split("_mean_")                    
        if (len(s) > 1):
            if (s[0] == '0'):
                s.remove('0')

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
            
    X_train_subset=X_train[features]
    X_valid_subset=X_valid[features]
    X_test_subset=X_test[features]
    
    lgb_train = lgb.Dataset(X_train_subset, y_train)
    lgb_eval = lgb.Dataset(X_valid_subset, y_valid, reference=lgb_train)

    model = lgb.train(lgb_pars,
            lgb_train,
            num_boost_round=rounds,
            valid_sets=lgb_eval,
            early_stopping_rounds=early,
            feval=gini_lgb,
            verbose_eval=100)

    p_train = model.predict(X_train_subset, num_iteration=model.best_iteration)            
    p_valid = model.predict(X_valid_subset, num_iteration=model.best_iteration)            
    p_test = model.predict(X_test_subset, num_iteration=model.best_iteration)   

    train_score = gini_normalizedc(y_train, p_train) 
    valid_score = gini_normalizedc(y_valid, p_valid)     

    return [train_score, valid_score, p_test]

########################### target_encode Another-kaggle #########################
########################### target_encode Another-kaggle #########################
########################### target_encode Another-kaggle #########################












########################### target_encode oliver-kaggle #########################
########################### target_encode oliver-kaggle #########################
########################### target_encode oliver-kaggle #########################

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# sample_submission.csv
# test.csv
# train.csv

# Target encoding with smoothing

# min_samples_leaf define a threshold where prior and target mean (for a given category value) have the same 
# weight. Below the threshold prior becomes more important and above mean becomes more important.

# How weight behaves against value counts is controlled by smoothing parameter

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
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
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

Testing with ps_car_11_cat

# reading data
trn_df = pd.read_csv("../input/train.csv", index_col=0)
sub_df = pd.read_csv("../input/test.csv", index_col=0)

# Target encode ps_car_11_cat
trn, sub = target_encode(trn_df["ps_car_11_cat"], 
                         sub_df["ps_car_11_cat"], 
                         target=trn_df.target, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)
trn.head(10)

id
7     0.038917
9     0.023708
13    0.030953
16    0.044688
17    0.026341
19    0.045348
20    0.022527
22    0.030300
26    0.033820
28    0.044770
Name: ps_car_11_cat_mean, dtype: float64

Scatter plot of category values vs target encoding

We see that the category values are not ordered

import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(trn_df["ps_car_11_cat"], trn)
plt.xlabel("ps_car_11_cat category values")
plt.ylabel("Noisy target encoding")

<matplotlib.text.Text at 0x7f6bcb8bc2b0>

Check AUC metric improvement after noisy encoding over 5 folds

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
f_cats = [f for f in trn_df.columns if "_cat" in f]
print("%20s   %20s | %20s" % ("", "Raw Categories", "Encoded Categories"))
for f in f_cats:
    print("%-20s : " % f, end="")
    e_scores = []
    f_scores = []
    for trn_idx, val_idx in folds.split(trn_df.values, trn_df.target.values):
        trn_f, trn_tgt = trn_df[f].iloc[trn_idx], trn_df.target.iloc[trn_idx]
        val_f, val_tgt = trn_df[f].iloc[trn_idx], trn_df.target.iloc[trn_idx]
        trn_tf, val_tf = target_encode(trn_series=trn_f, 
                                       tst_series=val_f, 
                                       target=trn_tgt, 
                                       min_samples_leaf=100, 
                                       smoothing=20,
                                       noise_level=0.01)
        f_scores.append(max(roc_auc_score(val_tgt, val_f), 1 - roc_auc_score(val_tgt, val_f)))
        e_scores.append(roc_auc_score(val_tgt, val_tf))
    print(" %.6f + %.6f | %6f + %.6f" 
          % (np.mean(f_scores), np.std(f_scores), np.mean(e_scores), np.std(e_scores)))

                             Raw Categories |   Encoded Categories
ps_ind_02_cat        :  0.506205 + 0.000645 | 0.508456 + 0.001132
ps_ind_04_cat        :  0.512617 + 0.000721 | 0.514476 + 0.001560
ps_ind_05_cat        :  0.520250 + 0.000760 | 0.534369 + 0.001934
ps_car_01_cat        :  0.528912 + 0.000736 | 0.552015 + 0.001517
ps_car_02_cat        :  0.531614 + 0.000786 | 0.531642 + 0.001340
ps_car_03_cat        :  0.539652 + 0.000782 | 0.539038 + 0.001555
ps_car_04_cat        :  0.536473 + 0.000532 | 0.536465 + 0.001234
ps_car_05_cat        :  0.530585 + 0.000432 | 0.530795 + 0.000676
ps_car_06_cat        :  0.515692 + 0.000817 | 0.542923 + 0.000841
ps_car_07_cat        :  0.522623 + 0.000671 | 0.522535 + 0.001586
ps_car_08_cat        :  0.520287 + 0.000602 | 0.521506 + 0.002040
ps_car_09_cat        :  0.504888 + 0.000700 | 0.525121 + 0.001698
ps_car_10_cat        :  0.500254 + 0.000058 | 0.501058 + 0.001909
ps_car_11_cat        :  0.512414 + 0.001029 | 0.574298 + 0.000581
########################### target_encode oliver-kaggle #########################
########################### target_encode oliver-kaggle #########################
########################### target_encode oliver-kaggle #########################









########################### target_encode kaggle comp #########################
########################### target_encode kaggle comp #########################
########################### target_encode kaggle comp #########################
import numpy as np
import pandas as pd
from rgf.sklearn import RGFClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from numba import jit
import time
import gc
import subprocess
import glob



# Compute gini

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini
    
    
    
# Funcitons from olivier's kernel
# https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,    # Revised to encode validation series
                  val_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
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
    
# Read data
train_df = pd.read_csv('../input/train.csv', na_values="-1") # .iloc[0:200,:]
test_df = pd.read_csv('../input/test.csv', na_values="-1")

# from olivier
train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
	"ps_reg_03",  #            : 1408.42 / shadow  511.15
	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
	"ps_ind_03",  #            : 1219.47 / shadow  230.55
	"ps_ind_15",  #            :  922.18 / shadow  242.00
	"ps_reg_02",  #            :  920.65 / shadow  267.50
	"ps_car_14",  #            :  798.48 / shadow  549.58
	"ps_car_12",  #            :  731.93 / shadow  293.62
	"ps_car_01_cat",  #        :  698.07 / shadow  178.72
	"ps_car_07_cat",  #        :  694.53 / shadow   36.35
	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15
	"ps_car_03_cat",  #        :  611.73 / shadow   50.67
	"ps_reg_01",  #            :  598.60 / shadow  178.57
	"ps_car_15",  #            :  593.35 / shadow  226.43
	"ps_ind_01",  #            :  547.32 / shadow  154.58
	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17
	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92
	"ps_car_06_cat",  #        :  398.02 / shadow  212.43
	"ps_car_04_cat",  #        :  376.87 / shadow   76.98
	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13
	"ps_car_09_cat",  #        :  214.12 / shadow   81.38
	"ps_car_02_cat",  #        :  203.03 / shadow   26.67
	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
	"ps_car_11",  #            :  173.28 / shadow   76.45
	"ps_car_05_cat",  #        :  172.75 / shadow   62.92
	"ps_calc_09",  #           :  169.13 / shadow  129.72
	"ps_calc_05",  #           :  148.83 / shadow  120.68
	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63
	"ps_car_08_cat",  #        :  120.87 / shadow   28.82
	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05
	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97
	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
	"ps_ind_14",  #            :   37.37 / shadow   16.65
]
# add combinations
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),  
    ('ps_reg_01', 'ps_car_04_cat'),
]

# Process data
id_test = test_df['id'].values
id_train = train_df['id'].values
y = train_df['target']

start = time.time()
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + "_plus_" + f2
    print('current feature %60s %4d in %5.1f'
          % (name1, n_c + 1, (time.time() - start) / 60), end='')
    print('\r' * 75, end='')
    train_df[name1] = train_df[f1].apply(lambda x: str(x)) + "_" + train_df[f2].apply(lambda x: str(x))
    test_df[name1] = test_df[f1].apply(lambda x: str(x)) + "_" + test_df[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
    train_df[name1] = lbl.transform(list(train_df[name1].values))
    test_df[name1] = lbl.transform(list(test_df[name1].values))

    train_features.append(name1)
    
X = train_df[train_features]
test_df = test_df[train_features]

f_cats = [f for f in X.columns if "_cat" in f]


y_valid_pred = 0*y
y_test_pred = 0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)

    
# Run CV

def run_rgf():
    model = RGFClassifier(
        max_leaf=1000,
        algorithm="RGF",  
        loss="Log",
        l2=0.01,
        sl2=0.01,
        normalize=False,
        min_samples_leaf=10,
        n_iter=None,
        opt_interval=100,
        learning_rate=.5,
        calc_prob="sigmoid",
        n_jobs=-1,
        memory_policy="generous",
        verbose=0
    )
    
    fit_model = model.fit( X_train, y_train )
    pred = fit_model.predict_proba(X_valid)[:,1]
    pred_test = fit_model.predict_proba(X_test)[:,1]
    try:
        subprocess.call('rm -rf /tmp/rgf/*', shell=True)
        print("Clean up is successfull")
        print(glob.glob("/tmp/rgf/*"))
    except Exception as e:
        print(str(e))
    
    return pred, pred_test
    

for i, (train_index, test_index) in enumerate(kf.split(train_df)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    X_test = test_df.copy()
    print( "\nFold ", i)
    
    # Enocode data
    for f in f_cats:
        X_train[f + "_avg"], X_valid[f + "_avg"], X_test[f + "_avg"] = target_encode(
                                                        trn_series=X_train[f],
                                                        val_series=X_valid[f],
                                                        tst_series=X_test[f],
                                                        target=y_train,
                                                        min_samples_leaf=200,
                                                        smoothing=10,
                                                        noise_level=0
                                                        )
    # Run model for this fold
    X_train = X_train.fillna(X_train.mean())
    X_valid = X_valid.fillna(X_valid.mean())
    X_test = X_test.fillna(X_test.mean())

    
        
    # Generate validation predictions for this fold
    pred, pred_test = run_rgf()
    
    
    print( "  Gini = ", eval_gini(y_valid, pred) )
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    y_test_pred += pred_test
    

    del X_test, X_train, X_valid, y_train

    gc.collect()
    gc.collect()
    gc.collect()
    
y_test_pred /= K  # Average test set predictions

print( "\nGini for full training set:" )
eval_gini(y, y_valid_pred)

# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('rgf_valid.csv', float_format='%.6f', index=False)

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('rgf_submit.csv', float_format='%.6f', index=False)
########################### target_encode kaggle comp #########################
########################### target_encode kaggle comp #########################
########################### target_encode kaggle comp #########################










########################### target_encode_std #########################
########################### target_encode_std #########################
########################### target_encode_std #########################
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode_std(train_series=None,
                      test_series=None,
                      target=None,
                      noise_level=0):
    assert len(train_series) == len(target)
    assert train_series.name == test_series.name

    temp = pd.concat([train_series, target], axis=1)
    # Compute target mean
    aggregated_values = temp.groupby(by=train_series.name)[target.name].agg(["mean", "count", np.std])
    total_std = np.std(target)
    aggregated_values["std"].fillna(total_std, inplace=True)

    # Compute smoothing
    smoothing_component = aggregated_values["count"] * total_std ** 2
    smoothing = smoothing_component / (aggregated_values["std"] ** 2 + smoothing_component)

    # Apply average function to all target data
    mean_total = target.mean()
    mean_values = mean_total * (1 - smoothing) + aggregated_values["mean"] * smoothing

    mean_values_dict = mean_values.rank(axis=0, method='first').to_dict()

    train_columns = train_series.replace(mean_values_dict).fillna(mean_total)
    test_columns = test_series.replace(mean_values_dict).fillna(mean_total)
    
    return add_noise(train_columns, noise_level), add_noise(test_columns, noise_level)
########################### target_encode_std #########################
########################### target_encode_std #########################
########################### target_encode_std #########################
