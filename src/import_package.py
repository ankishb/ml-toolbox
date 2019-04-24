
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os, gc
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_rows",200)
pd.set_option("display.max_columns",100)


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import Pool, CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb