
1 selected

Skip to content
Using Gmail with screen readers
Conversations
2.38 GB (15%) of 15 GB used
Manage
Terms · Privacy · Program Policies
Last account activity: 14 minutes ago
Details

# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os, gc
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_rows",200)
pd.set_option("display.max_columns",100)


# In[ ]:


train_df = pd.read_csv('store_data/new_feat/train_df_feat.csv')
test_df  = pd.read_csv('store_data/new_feat/test_df_feat.csv')
sub      = pd.read_csv('sample_submission_24jSKY6.csv')
info_df  = pd.read_excel('train_aox2Jxw/Data Dictionary.xlsx')
train_df.shape, test_df.shape, sub.shape, info_df.shape


# In[ ]:


train_df1 = pd.read_csv('train_aox2Jxw/train.csv')
test_df1  = pd.read_csv('test_bqCt9Pv.csv')

train_df1 = train_df1[['UniqueID']]
test_df1 = test_df1[['UniqueID']]

train_df1.shape, test_df1.shape


# In[2]:


train_df = pd.read_csv('train_aox2Jxw/train.csv')
test_df  = pd.read_csv('test_bqCt9Pv.csv')
sub      = pd.read_csv('sample_submission_24jSKY6.csv')
info_df  = pd.read_excel('train_aox2Jxw/Data Dictionary.xlsx')

train_test = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

train_test.shape, train_df.shape, test_df.shape, sub.shape, info_df.shape


# # Only `Employment.Type` have `NaN` values

# ## Add flag to diffrentiate b/w train and test

# In[3]:


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
    


# In[4]:


def remove_and_fill_outlier(df, flag):
    """fill outlier with the nan values"""
    """
    Args:
        df  : feature
        flag: string ['lower','upper','both']
    return: array containing new feature
    """
    lower_bound = np.percentile(df, q=1)
    upper_bound = np.percentile(df, q=99)
    print("low: ", np.round(lower_bound,2), "   high: ", np.round(upper_bound,2))
    if flag == 'upper':
        new = np.where(df>upper_bound, np.nan, df)
    elif flag == 'lower':
        new = np.where(df<lower_bound, np.nan, df)
    else: # when both are selected
        new = np.where(df>upper_bound, np.nan, df)
        new = np.where(new<lower_bound, np.nan, new)
    print("null count: ", pd.DataFrame(new).isnull().sum().values)
    return new

def get_quantile(df, col, q1, q2):
    lower_bound = np.percentile(df[col], q=q1)
    upper_bound = np.percentile(df[col], q=q2)
    print("low: ", lower_bound)
    print("high: ", upper_bound)
    
def remove_outlier(df, col, flag):
    """fill outlier with the nan values"""
    """
    Args:
        df  : feature
        flag: string ['lower','upper','both']
    return: array containing new feature
    """
    lower_bound = np.percentile(df[col], q=1)
    upper_bound = np.percentile(df[col], q=99)
    print("low: ", np.round(lower_bound,2), "   high: ", np.round(upper_bound,2))
    print("check shape: ", df.shape[0], "==>", end=" ")
    if flag == 'upper':
        df = df[df[col]<upper_bound]
    elif flag == 'lower':
        df = df[df[col]>lower_bound]
    else: # when both are selected
        df = df[df[col]<upper_bound]
        df = df[df[col]>lower_bound]
    print(df.shape[0])
    return d


# # Handling `12` catgorical variables

# In[5]:


train_test['train_flag'] = 0
train_test['train_flag'].iloc[:train_df.shape[0]] = 1
train_test.train_flag.value_counts()


# In[6]:


train_test['Date.of.Birth'] = train_test['Date.of.Birth'].apply(
    lambda x: str(x)[:-2]+'20'+str(x)[-2:] if str(x)[-2:] == '00' \
    else str(x)[:-2]+'19'+str(x)[-2:])
train_test['date_of_birth']  = pd.to_datetime(pd.Series(train_test['Date.of.Birth']))
train_test.drop('Date.of.Birth', axis=1, inplace=True)


# In[7]:


train_test['DisbursalDate_new'] = pd.to_datetime(
    pd.Series(train_test['DisbursalDate'].apply(lambda x: str(x)[:-3])), 
    format="%d-%m")

train_test.DisbursalDate_new = train_test.DisbursalDate_new.apply(
    lambda dt: dt.replace(year=2018))

train_test['disbursal_week']  = train_test['DisbursalDate_new'].dt.week
train_test['disbursal_day']   = train_test['DisbursalDate_new'].dt.day
train_test['disbursal_month'] = train_test['DisbursalDate_new'].dt.month

# train_test.DisbursalDate_new.unique()


# In[8]:


train_test['age(in years)'] = ((train_test['DisbursalDate_new'] -                     train_test['date_of_birth'])/365) / np.timedelta64(1, 'D')
train_test['age(in years)'] = train_test['age(in years)'].astype('int')
print(train_test['age(in years)'][:5])

train_test['age(in month)'] = ((train_test['DisbursalDate_new'] -                     train_test['date_of_birth'])/30) / np.timedelta64(1, 'D')
train_test['age(in month)'] = train_test['age(in years)'].astype('int')
print(train_test['age(in month)'][:5])


# In[9]:


train_test['credit_hist_year'] = train_test['CREDIT.HISTORY.LENGTH'].apply(lambda x: x.split(' ')[0][:-3])
train_test['credit_hist_month']= train_test['CREDIT.HISTORY.LENGTH'].apply(lambda x: x.split(' ')[1][:-3])

train_test['credit_hist_year'] = train_test['credit_hist_year'].astype('int')
train_test['credit_hist_month']= train_test['credit_hist_month'].astype('int')

train_test['credit_hist_total_month']= train_test['credit_hist_month'] +                                         train_test['credit_hist_year']*12

train_test['loan_tenure_year'] = train_test['AVERAGE.ACCT.AGE'].apply(lambda x: x.split(' ')[0][:-3])
train_test['loan_tenure_month']= train_test['AVERAGE.ACCT.AGE'].apply(lambda x: x.split(' ')[1][:-3])

train_test['loan_tenure_year'] = train_test['loan_tenure_year'].astype('int')
train_test['loan_tenure_month']= train_test['loan_tenure_month'].astype('int')

train_test['loan_tenure_total_month']= train_test['loan_tenure_month'] +                                         train_test['loan_tenure_year']*12

# train_test.drop(['AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH'], axis=1, inplace=True)


# In[10]:


train_test['Employment.Type'].fillna('Self employed', inplace=True)


# In[11]:


col = 'PRI.NO.OF.ACCTS'
new_col = 'pri_no_of_accts'

bins = pd.IntervalIndex.from_tuples([(-1,0),(0,1),(1,2),(2,3),(3,4),(4,8),(8,15),(15,1000)])
train_test[new_col] = pd.cut(train_test[col], bins)
print(train_test[new_col].value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_test[new_col] = le.fit_transform(train_test[new_col])
print(le.classes_)

train_test.drop(col, axis=1, inplace=True)
train_test.groupby([new_col,'loan_default'])['loan_default'].count().unstack()


# In[12]:


col = 'PRI.ACTIVE.ACCTS'

bins = pd.IntervalIndex.from_tuples([(-1, 0),(0, 1),(1,3),(3,6),(6,10),(10,500)])
train_test['no_of_acc'] = pd.cut(train_test[col], bins)
print(train_test['no_of_acc'].value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_test['no_of_acc'] = le.fit_transform(train_test['no_of_acc'])
print(le.classes_)

train_test.drop(col, axis=1, inplace=True)
train_test.groupby(['no_of_acc','loan_default'])['loan_default'].count().unstack()


# In[13]:


col = 'PRI.OVERDUE.ACCTS'
new_col = 'no_of_acc_overdue'
bins = pd.IntervalIndex.from_tuples([(-1, 0),(0, 1),(1,2),(2,5),(5,500)])
train_test[new_col] = pd.cut(train_test[col], bins)
print(train_test[new_col].value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_test[new_col] = le.fit_transform(train_test[new_col])
print(le.classes_)
print(train_test.groupby([new_col,'loan_default'])['loan_default'].count().unstack())

train_test.drop(col, axis=1, inplace=True)


# In[14]:


train_test.drop('MobileNo_Avl_Flag', axis=1, inplace=True)


# In[15]:


# bureau description and corresponding score
mapping = {
        'No Bureau History Available'                            :'not_enough_info',# 00
        'Not Scored: More than 50 active Accounts found'         :'not_enough_info',# 11
        'Not Scored: No Activity seen on the customer (Inactive)':'not_enough_info',# 16
        'Not Scored: No Updates available in last 36 months'     :'not_enough_info',# 18 
        'Not Scored: Not Enough Info available on the customer'  :'not_enough_info',# 17
        'Not Scored: Only a Guarantor'                           :'not_enough_info',# 14
        'Not Scored: Sufficient History Not Available'           :'not_enough_info' # 15
       }

train_test['Bureau_desc'] = train_test['PERFORM_CNS.SCORE.DESCRIPTION'].replace(mapping)

use_index = train_test[['PERFORM_CNS.SCORE.DESCRIPTION','PERFORM_CNS.SCORE']][
    train_test['Bureau_desc'] == 'not_enough_info']['PERFORM_CNS.SCORE'].index
train_test['bureau_score'] = train_test['PERFORM_CNS.SCORE']
train_test.loc[use_index,'bureau_score'] = 0


train_test.drop(['PERFORM_CNS.SCORE.DESCRIPTION','PERFORM_CNS.SCORE'], 
                axis=1, inplace=True)


# In[16]:


col = 'NEW.ACCTS.IN.LAST.SIX.MONTHS'
new_col = 'new_acc_past_month'

bins = pd.IntervalIndex.from_tuples([(-1,0),(0,1),(1,2),(2,3),(3,4),(4,50)])
train_test[new_col] = pd.cut(train_test[col], bins)
print(train_test[new_col].value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_test[new_col] = le.fit_transform(train_test[new_col])
print(le.classes_)

train_test.groupby([new_col,'loan_default'])['loan_default'].count().unstack()
train_test.drop(col, axis=1, inplace=True)


# # numeric columns

# In[17]:


def print_it(col):
    print(col)
    print("train-test:")
    get_quantile(train_test,col,q1,q2)
    print("train_df  :")
    get_quantile(train_df,col,q1,q2)
    print("*"*30)

q1 = 1
q2 = 99

col = 'PRI.SANCTIONED.AMOUNT'; print_it(col)
col = 'PRI.DISBURSED.AMOUNT'; print_it(col)
col = 'PRI.SANCTIONED.AMOUNT'; print_it(col)
col = 'PRI.DISBURSED.AMOUNT'; print_it(col)
col = 'PRI.CURRENT.BALANCE'; print_it(col)
col = 'SEC.CURRENT.BALANCE'; print_it(col)
col = 'SEC.DISBURSED.AMOUNT'; print_it(col)
col = 'PRIMARY.INSTAL.AMT'; print_it(col)
col = 'SEC.CURRENT.BALANCE'; print_it(col)


# In[18]:


train_test.shape, train_df.shape, test_df.shape, train_df.shape[0]+test_df.shape[0]


# In[19]:


def replace_outliers(df, low, high):
    """
    Args:
        df  : array
        low : 01% range
        high: 99% range
    return:
        an array, where outliers are replaced by the end ranges
    """
    df = np.where(df>high, high, df)
    df = np.where(df<low, low, df)
    print(df)
    
x = [1,2,3,4,54,5,7,78,9,9,33,5,6,9]
replace_outliers(np.array(x), 3,7)


# In[20]:


df = pd.DataFrame(data=x, columns=['x'])
print(x)
df[df.x > 7] = 7
df[df.x < 3] = 3
df


# In[21]:


def count_negative(df, col):
    print(col, "==", df[df[col]<0].shape[0])
print("count negative numbers")

col = 'PRI.SANCTIONED.AMOUNT'; count_negative(train_test, col)
col = 'PRI.DISBURSED.AMOUNT'; count_negative(train_test, col)
col = 'PRI.SANCTIONED.AMOUNT'; count_negative(train_test, col)
col = 'PRI.DISBURSED.AMOUNT'; count_negative(train_test, col)
col = 'PRI.CURRENT.BALANCE'; count_negative(train_test, col)
col = 'SEC.CURRENT.BALANCE'; count_negative(train_test, col)
col = 'SEC.DISBURSED.AMOUNT'; count_negative(train_test, col)
col = 'PRIMARY.INSTAL.AMT'; count_negative(train_test, col)
col = 'SEC.CURRENT.BALANCE'; count_negative(train_test, col)


# In[22]:


def print_it(col):
    print(col)
#     print("train-test:")
    get_quantile(train_test,col,q1,q2)
#     print("train_df  :")
#     get_quantile(train_df,col,q1,q2)
    print("*"*30)

q1 = 1
q2 = 99

col = 'PRI.SANCTIONED.AMOUNT'; print_it(col)
col = 'PRI.DISBURSED.AMOUNT'; print_it(col)
col = 'PRI.SANCTIONED.AMOUNT'; print_it(col)
col = 'PRI.DISBURSED.AMOUNT'; print_it(col)
col = 'PRI.CURRENT.BALANCE'; print_it(col)
col = 'SEC.CURRENT.BALANCE'; print_it(col)
col = 'SEC.DISBURSED.AMOUNT'; print_it(col)
col = 'PRIMARY.INSTAL.AMT'; print_it(col)
col = 'SEC.CURRENT.BALANCE'; print_it(col)


# In[23]:


col = 'PRI.SANCTIONED.AMOUNT'

print(train_test.shape, "==>", end=" ")
train_test[train_test[col] < 0][col] = 0
train_test[train_test[col] > 3363234][col] = 3363234
print(train_test.shape)

col = 'PRI.DISBURSED.AMOUNT'

print(train_test.shape, "==>", end=" ")
train_test[train_test[col] < 0][col] = 0
train_test[train_test[col] > 3366855][col] = 3366855
print(train_test.shape)

col = 'PRI.CURRENT.BALANCE'

print(train_test.shape, "==>", end=" ")
train_test[train_test[col]<0][col] = 0
train_test[train_test[col]>2835998][col] = 2835998
print(train_test.shape)

col = 'SEC.CURRENT.BALANCE'

print(train_test.shape, "==>", end=" ")
train_test[train_test[col]<0][col] = 0
train_test[train_test[col]>6226][col] = 6226
print(train_test.shape)

col = 'SEC.DISBURSED.AMOUNT'

print(train_test.shape, "==>", end=" ")
train_test[train_test[col]<0][col] = 0
train_test[train_test[col]>30397][col] = 30397
print(train_test.shape)

col = 'SEC.CURRENT.BALANCE'

print(train_test.shape, "==>", end=" ")
train_test[train_test[col]<0][col] = 0 
train_test[train_test[col]>6226.5][col] = 6226.5
print(train_test.shape)

col = 'PRIMARY.INSTAL.AMT'

print(train_test.shape, "==>", end=" ")
train_test[train_test[col]<0][col] = 0
train_test[train_test[col]>240000][col] = 240000
print(train_test.shape)


# In[24]:


col1 = 'PRI.SANCTIONED.AMOUNT'
col2 = 'PRI.DISBURSED.AMOUNT'
train_test['pay_by_yourself'] = train_test[col2] - train_test[col1]

new_col1 = 'obtained_amount_per_month'
new_col2 = 'obtained_amount_per_year'
train_test[new_col1] = train_test[col2]/(1 + train_test['loan_tenure_total_month'])
train_test[new_col2] = train_test[col2]/(1 + train_test['loan_tenure_year'])


# In[25]:


print(train_test.shape[0], "==>", end=" ")
drop_cols = ['SEC.OVERDUE.ACCTS', 'SEC.ACTIVE.ACCTS', 'SEC.NO.OF.ACCTS', 'VoterID_flag', 
             'Driving_flag', 'Passport_flag', 'SEC.INSTAL.AMT']
train_test.drop(drop_cols, axis=1, inplace=True)
print(train_test.shape[0])


# In[26]:


############## Simple aggregation w/o time compomnenet###########
print(train_test.shape, "==>", end=" ")

####################### manufacturer_id Mean #######################
branch_gps = train_test.groupby(['manufacturer_id'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['manufac_disbursed_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['manufacturer_id'],how='left')

####################### State_ID Mean #######################
branch_gps = train_test.groupby(['State_ID'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['state_disbursed_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['State_ID'],how='left')

####################### branch_id Mean #######################
branch_gps = train_test.groupby(['branch_id'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['branch_disbursed_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['branch_id'],how='left')

print(train_test.shape)


# # Aggregation with time compomnenet
#     - It have different distribution for train and test

# In[27]:


############## Aggregation with time compomnenet ###########

print(train_test.shape, "==>", end=" ")

####################### manufacturer_id Mean #######################
branch_gps = train_test.groupby(['manufacturer_id','disbursal_week'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['manufac_week_disbursed_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['manufacturer_id','disbursal_week'],how='left')

####################### State_ID Mean #######################
branch_gps = train_test.groupby(['State_ID','disbursal_week'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['state_week_disbursed_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['State_ID','disbursal_week'],how='left')

####################### branch_id Mean #######################
branch_gps = train_test.groupby(['branch_id','disbursal_week'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['branch_week_disbursed_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['branch_id','disbursal_week'],how='left')

print(train_test.shape)


# ## aggregation date wise

# In[28]:


train_test1 = train_test.copy()
############## Simple aggregation w/o time compomnenet###########
print(train_test.shape, "==>", end=" ")

####################### manufacturer_id Mean #######################
branch_gps = train_test.groupby(['DisbursalDate_new'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['manufac_disbursedDate_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['DisbursalDate_new'],how='left')

####################### State_ID Mean #######################
branch_gps = train_test.groupby(['DisbursalDate_new'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['state_disbursedDate_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['DisbursalDate_new'],how='left')

####################### branch_id Mean #######################
branch_gps = train_test.groupby(['DisbursalDate_new'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['branch_disbursedDate_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['DisbursalDate_new'],how='left')

print(train_test.shape)


# In[29]:


############## Aggregation with time compomnenet ###########

print(train_test.shape, "==>", end=" ")

####################### manufacturer_id Mean #######################
branch_gps = train_test.groupby(['manufacturer_id','DisbursalDate_new'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['manufac_date_disbursed_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['manufacturer_id','DisbursalDate_new'],how='left')

####################### State_ID Mean #######################
branch_gps = train_test.groupby(['State_ID','DisbursalDate_new'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['state_date_disbursed_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['State_ID','DisbursalDate_new'],how='left')

####################### branch_id Mean #######################
branch_gps = train_test.groupby(['branch_id','DisbursalDate_new'])[
    'disbursed_amount'].aggregate(['mean'])
branch_gps.columns = ['branch_date_disbursed_mean']
train_test = pd.merge(train_test, branch_gps, 
                      on=['branch_id','DisbursalDate_new'],how='left')

print(train_test.shape)


# In[31]:


stat_cols = ['PRI.CURRENT.BALANCE','PRI.DISBURSED.AMOUNT','PRI.SANCTIONED.AMOUNT',
             'PRIMARY.INSTAL.AMT','asset_cost','disbursed_amount']

print(train_test.shape, "==>", end=" ")

train_test['stat_mean']   = train_test[stat_cols].mean(axis=1)
train_test['stat_median'] = train_test[stat_cols].median(axis=1)
train_test['stat_skew']   = train_test[stat_cols].skew(axis=1)
train_test['stat_std']    = train_test[stat_cols].std(axis=1)
print(train_test.shape)


# In[32]:


assert train_test.shape[0] == train_df.shape[0]+test_df.shape[0]
print("shape are equal")


# In[33]:


branch_gping = train_test.groupby(['DisbursalDate','branch_id','Employee_code_ID'])[
    'disbursed_amount', 'ltv','asset_cost'].mean()
branch_gping.columns = ['date_branch_emp_disbursed_amount', 'date_branch_emp_ltv', 
                        'date_branch_emp_asset_cost']

print(train_test.shape, "==>", end=" ")
train_test = pd.merge(train_test, branch_gping, 
                      on=['DisbursalDate','branch_id','Employee_code_ID'],
                      how='left')
print(train_test.shape)


# In[33]:


train_test.to_csv('store_data/train_test_save7.csv', index=None)


# In[34]:


drop_cols = ['date_of_birth', 'Employee_code_ID','DisbursalDate',
             'CREDIT.HISTORY.LENGTH', 'AVERAGE.ACCT.AGE', 'DisbursalDate_new']

train_test.drop(drop_cols, axis=1, inplace=True)
train_test.head()


# In[35]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
col = 'Bureau_desc'
train_test[col] = le.fit_transform(train_test[col])
print(le.classes_)
col = 'Employment.Type'
train_test[col] = le.fit_transform(train_test[col])
print(le.classes_)


# In[36]:


tr_shape = train_df.shape[0]
ts_shape = test_df.shape[0]

del train_df, test_df
gc.collect()

train_test.head()


# In[37]:


train_df = train_test.iloc[:tr_shape, :]
test_df = train_test.iloc[tr_shape:, :]
test_df.reset_index(drop=True, inplace=True)

tr_unique_ids = train_df.UniqueID
ts_unique_ids = test_df.UniqueID

train_df.shape, test_df.shape


# In[38]:


target = train_df['loan_default']

train_df.drop(['loan_default','train_flag','UniqueID'], axis=1, inplace=True)
test_df.drop(['loan_default','train_flag','UniqueID'], axis=1, inplace=True)

train_df.shape, test_df.shape


# In[37]:


train_test.dtypes


# In[39]:


print(train_df.shape, test_df.shape, "==>", end=" ")
train_df.drop(['disbursal_week', 'disbursal_day'], axis=1, inplace=True)
test_df.drop(['disbursal_week', 'disbursal_day'], axis=1, inplace=True)
print(train_df.shape, test_df.shape)


# In[49]:


tr_unique_ids = train_df1.UniqueID
ts_unique_ids = test_df1.UniqueID

target = train_df['loan_default']

train_df.drop('loan_default', axis=1, inplace=True)

del train_df1, test_df1
gc.collect()

train_df.shape, test_df.shape, target.shape


# In[43]:


cols = train_df.apply(lambda x: pd.Series.value_counts(x).shape[0]).index
values = train_df.apply(lambda x: pd.Series.value_counts(x).shape[0]).values
plt.plot(cols, values, 'o')

complete_df = pd.concat([train_df, test_df], axis=0)

for col,value in zip(cols,values):
    if value>10 or value<300:
        q1 = 15
        complete_df[col] = pd.qcut(complete_df[col], q=q1, 
                                  retbins=False, duplicates='drop')
    elif value>300 or value<10000:
        q1 = 50
        complete_df[col] = pd.qcut(complete_df[col], q=q1, 
                                  retbins=False, duplicates='drop')
    elif value>10000:# or value<10000:
        q1 = 100
        complete_df[col] = pd.qcut(complete_df[col], q=q1, 
                                  retbins=False, duplicates='drop')
    else:
        pass
    
train_df = complete_df.iloc[:train_df.shape[0],:]
test_df  = complete_df.iloc[train_df.shape[0]:,:]
del complete_df
gc.collect()


# In[45]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import Pool, CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb


# In[46]:


def train_lgb_model(X_train, y_train, X_valid, y_valid, features, param, X_test, num_round):
    """
    Args:
        X_train, X_valid: training and valid data
        y_train, y_valid: training and valid target
        X_test: test-data
        features: training features
    Return:
        oof-pred, test_preds model, model_imp
    """
    _train = lgb.Dataset(X_train[features], label=y_train, feature_name=list(features))
    _valid = lgb.Dataset(X_valid[features], label=y_valid,feature_name=list(features))
    
    clf = lgb.train(param, _train, num_round, 
                    valid_sets = [_train, _valid], 
                    verbose_eval=200, 
                    early_stopping_rounds = 25)                  
    
    oof = clf.predict(X_valid[features], num_iteration=clf.best_iteration)
    test_pred = clf.predict(X_test[features], num_iteration=clf.best_iteration)
    
    lgb_imp = pd.DataFrame(data=[clf.feature_name(), list(clf.feature_importance())]).T
    lgb_imp.columns = ['feature','imp']
    
    return oof, test_pred, clf, lgb_imp
    
    
def train_xgb_model(X_train, y_train, X_valid, y_valid, features, param, X_test, 
                    num_round):
    """
    Args:
        X_train, X_valid: training and valid data
        y_train, y_valid: training and valid target
        X_test: test-data
        features: training features
    Return:
        oof-pred, test_preds, model, model_imp
    """
    _train = xgb.DMatrix(X_train[features], label=y_train, feature_names=list(features))
    _valid = xgb.DMatrix(X_valid[features], label=y_valid,feature_names=list(features))
    
    watchlist = [(_valid, 'valid')]
    clf = xgb.train(dtrain=_train, 
                    num_boost_round=num_round, 
                    evals=watchlist,
                    early_stopping_rounds=25, 
                    verbose_eval=200, 
                    params=param)
    
    valid_frame = xgb.DMatrix(X_valid[features],feature_names=list(features))
    oof  = clf.predict(valid_frame, ntree_limit=clf.best_ntree_limit)


    test_frame = xgb.DMatrix(X_test[features],feature_names=list(features))
    test_pred = clf.predict(test_frame, ntree_limit=clf.best_ntree_limit)

    
    xgb_imp = pd.DataFrame(data=[list(clf.get_fscore().keys()), 
                                 list(clf.get_fscore().values())]).T
    xgb_imp.columns = ['feature','imp']
    xgb_imp.imp = xgb_imp.imp.astype('float')
    
    return oof, test_pred, clf, xgb_imp


def train_cat_model(X_train, y_train, X_valid, y_valid, features, param, X_test, 
                    num_round):
    """
    Args:
        X_train, X_valid: training and valid data
        y_train, y_valid: training and valid target
        X_test: test-data
        features: training features
    Return:
        oof-pred, test_preds, model, model_imp
    """
    param['iterations'] = num_round
    
    _train = Pool(X_train[features], label=y_train)#, cat_features=cate_features_index)
    _valid = Pool(X_valid[features], label=y_valid)#, cat_features=cate_features_index)

    watchlist = [_train, _valid]
    clf = CatBoostClassifier(**param)
    clf.fit(_train, 
            eval_set=watchlist, 
            verbose=200,
            use_best_model=True)
        
    oof  = clf.predict_proba(X_valid[features])[:,1]
    test_pred  = clf.predict_proba(X_test[features])[:,1]

    cat_imp = pd.DataFrame(data=[clf.feature_names_, 
                                 list(clf.feature_importances_)]).T
    cat_imp.columns = ['feature','imp']
    
    return oof, test_pred, clf, cat_imp


# In[47]:


# def run_xgb(splits, file_path, train_df, target, test_df, test_id, sub, depth):
def run_cv_xgb(train_df, target, test_df, depth):

    features = train_df.columns
    params = {
        'eval_metric'     : 'auc',
        'seed'            : 1337,
        'eta'             : 0.05,
        'subsample'       : 0.7,
        'colsample_bytree': 0.5,
        'silent'          : 1,
        'nthread'         : 4,
        'Scale_pos_weight': 3.607,
        'objective'       : 'binary:logistic',
        'max_depth'       : depth,
        'alpha'           : 0.05
    }
    
    n_splits = 3
    random_seed = 1234
    feature_imp = pd.DataFrame()
    
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_xgb = np.zeros(len(train_df))
    predictions = np.zeros((len(test_df),n_splits))
    clfs = []
##########################
    for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):
        print(train_index.shape, valid_index.shape)
        print("Fold {}".format(fold_))
    
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
        features = X_train.columns
        

        num_rounds = 10000
        oof, test_pred, clf, xgb_imp = train_xgb_model(X_train, y_train, 
                                                       X_valid, y_valid, 
                                                       features, params, 
                                                       test_df, num_rounds)
        
        xgb_imp['fold'] = fold_
        feature_imp = pd.concat([feature_imp, xgb_imp], axis=0)
    
        oof_xgb[valid_index] = oof
        predictions[:,fold_] = test_pred
        clfs.append(clf)
        
        score = roc_auc_score(y_valid, oof)
        print( "  auc = ", score )
        print("="*60)
    
    return clfs, feature_imp, oof_xgb, predictions


# In[48]:


def run_cv_cat(train_df, target, test_df, depth):

    params = {
        'loss_function'         : "Logloss", 
        'eval_metric'           : "AUC",
        'random_strength'       : 1.5,
        'border_count'          : 128,
        'scale_pos_weight'      : 3.507,
        'depth'                 : depth, 
        'early_stopping_rounds' : 50,
        'random_seed'           : 1337,
        'task_type'             : 'CPU', 
#         'subsample'             = 0.7, 
        'iterations'            : 10000, 
        'learning_rate'         : 0.09,
        'thread_count'          : 4
    }


    ##########################
    n_splits = 3
    random_seed = 1234
    feature_imp = pd.DataFrame()
    
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_cat = np.zeros(len(train_df))
    predictions = np.zeros((len(test_df),n_splits))
    clfs = []
##########################
    for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):
        print(train_index.shape, valid_index.shape)
        print("Fold {}".format(fold_))
    
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
        features = X_train.columns
        
        num_rounds = 10000
        oof, test_pred, clf, cat_imp = train_cat_model(X_train, y_train, 
                                                       X_valid, y_valid, 
                                                       features, params, 
                                                       test_df, num_rounds)
    
        oof_cat[valid_index] = oof
        predictions[:,fold_] = test_pred
        
        cat_imp['fold'] = fold_
        feature_imp = pd.concat([feature_imp, cat_imp], axis=0)
        clfs.append(clf)
        
        score = roc_auc_score(y_valid, oof)
        print( "  auc = ", score )
        print("="*60)
    
    return clfs, feature_imp, oof_cat, predictions


# In[49]:


def run_cv_lgb(train_df, target, test_df, leaves=None):

    param = {
        'bagging_freq'           : 5,
        'bagging_fraction'       : 0.33,
        'boost_from_average'     : 'false',
        'boost'                  : 'gbdt',
        'feature_fraction'       : 0.3,
        'learning_rate'          : 0.01,
        'max_depth'              : -1,
        'metric'                 : 'auc',
        'min_data_in_leaf'       : 100,
#         'min_sum_hessian_in_leaf': 10.0,
        'num_leaves'             : 30,
        'num_threads'            : 4,
        'tree_learner'           : 'serial',
        'objective'              : 'binary',
        'verbosity'              : 1,
    #     'lambda_l1'              : 0.001,
        'lambda_l2'              : 0.1
    }   
    if leaves is not None:
        param['num_leaves'] = leaves
        print("using leaves: ", param['num_leaves'])

    random_seed = 1234
    n_splits = 3
    num_round = 10000
    feature_imp = pd.DataFrame()
    
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_lgb = np.zeros(len(train_df))
    predictions = np.zeros((len(test_df),n_splits))

    clfs = []
    
    for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):
        print(train_index.shape, valid_index.shape)
        print("Fold {}".format(fold_))
    
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]
        features = X_train.columns
        
#         X_train.drop(['disbursal_week','disbursal_day'], axis=1, inplace=True)
#         X_valid.drop(['disbursal_week','disbursal_day'], axis=1, inplace=True)

        num_round = 10000
        oof, test_pred, clf, lgb_imp = train_lgb_model(X_train, y_train, 
                                                       X_valid, y_valid, 
                                                       features, param, 
                                                       test_df, num_round)
        lgb_imp['fold'] = fold_
        feature_imp = pd.concat([feature_imp, lgb_imp], axis=0)
    
        oof_lgb[valid_index] = oof
        predictions[:,fold_] = test_pred
        clfs.append(clf)
        
        score = roc_auc_score(y_valid, oof)
        print( "  auc = ", score )
        print("="*60)
    
    return clfs, feature_imp, oof_lgb, predictions


# In[50]:



def reduce_mem_usage_wo_print(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    print("="*30)
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory {:.2f} MB'.format(start_mem), "==>", end=" ")
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                    # print(col, "== int8")
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                    # print(col, "== int16")
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    # print(col, "== int32")
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                    # print(col, "== int64")
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                    # print(col, "== float16")
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    # print(col, "== float32")
                else:
                    df[col] = df[col].astype(np.float64)
                    # print(col, "== float64")
        #else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(' {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    print("="*30)
    return 

def make_prediction(file_path, df, test_ids, sub_df):
    """
    Args:
        file_path: file-name with base-path as "submission"
        df: array with shape (test_df.shape[0], cv_fold)
        test_ids: test_ids
        sub_df: submission data-frame
        
    Return:
        output a file with given name
        
    Example: 
    >>> make_prediction(file_path, predictions, ts_unique_ids, sub)
    """
    predictions = np.mean(df, axis=1)
    sub_df = pd.DataFrame({"ID_code":test_ids})
    sub_df["target"] = predictions
    sub_df.columns = sub.columns

    sub_df.to_csv('submission/stacking/{}.csv'.format(file_path), index=None)
    print("successfully saved")
#     print(sub_df.shape)
#     print(sub_df.sample(10))


# In[ ]:


# # def run_cv_lgb(file_path, train_df, target, test_df, test_ids, sub, leaves=None):
# def run_cv_fold(train_df, target):
    
#     random_seed = 1234
#     n_splits = 3
    
#     folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    
#     for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):
#         print(train_index.shape, valid_index.shape)
#         print("Fold {}".format(fold_))
    
#         y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
#         X_train, X_valid = train_df.iloc[train_index,:], train_df.iloc[valid_index,:]


# In[56]:


reduce_mem_usage_wo_print(train_df)
reduce_mem_usage_wo_print(test_df)


# In[51]:


model_lgb1, imp_lgb1, oof_lgb1, pred_lgb1 = run_cv_lgb(train_df, target, test_df)
make_prediction("lgb11", pred_lgb1, ts_unique_ids, sub)
np.save('submission/stacking/oof_lgb11.npy',oof_lgb1)
del oof_lgb1, pred_lgb1
gc.collect()


# In[ ]:


# cols = train_df.apply(lambda x: pd.Series.value_counts(x).shape[0]).index
# values = train_df.apply(lambda x: pd.Series.value_counts(x).shape[0]).values
# plt.plot(cols, values, 'o')

# complete_df = pd.concat([train_df, test_df], axis=0)

# for col,value in zip(cols,values):
#     if value>10 or value<300:
#         q1 = 15
#         complete_df[col] = pd.qcut(complete_df[col], q=q1, retbins=False, duplicates='drop')
#     elif value>300:
#         q1 = 50
#         complete_df[col] = pd.qcut(complete_df[col], q=q1, retbins=False, duplicates='drop')
#     else:
#         pass
    
# train_df1 = complete_df.iloc[:train_df.shape[0],:]
# test_df1  = complete_df.iloc[train_df.shape[0]:,:]
# del complete_df_df
# gc.collect()


# In[74]:



# model_lgb2, imp_lgb2, oof_lgb2, pred_lgb2 = run_cv_lgb(train_df1, target, test_df1)
# make_prediction("lgb12", pred_lgb2, ts_unique_ids, sub)
# np.save('submission/stacking/oof_lgb12.npy',oof_lgb2)
# del oof_lgb2, pred_lgb2
# gc.collect()


# In[75]:


imp_lgb2.imp = imp_lgb2.imp.astype('float')
imp_lgb2 = imp_lgb2.groupby(['feature'])['imp'].mean()
imp_lgb2 = pd.DataFrame(data=[imp_lgb2.index, imp_lgb2.values]).T
imp_lgb2.columns=['feature','imp']
imp_lgb2 = imp_lgb2.sort_values(by='imp')
plt.figure(figsize=(12,15))
plt.barh(imp_lgb2.feature, imp_lgb2.imp)


# In[ ]:


# model_lgb2, imp_lgb2, oof_lgb2, pred_lgb2 = run_cv_lgb(train_df, target, test_df)
# make_prediction("lgb12", pred_lgb2, ts_unique_ids, sub)
# np.save('submission/stacking/oof_lgb12.npy',oof_lgb2)
# del oof_lgb2, pred_lgb2
# gc.collect()


# In[12]:


# model_lgb1, imp_lgb1, oof_lgb1, pred_lgb1 = run_cv_lgb(train_df, target, test_df)
# make_prediction("lgb1", pred_lgb1, ts_unique_ids, sub)
# np.save('submission/stacking/oof_lgb1.npy',oof_lgb1)
# del oof_lgb1, pred_lgb1
# gc.collect()


# In[56]:


# # def (train_df, target, test_df, depth):
# model_xgb1, imp_xgb1, oof_xgb1, pred_xgb1 = run_cv_xgb(train_df.astype('int'), target,
#                                                        test_df.astype('int'), 4)
# #     train_df.drop('Current_pincode_ID', axis=1), target, 
# #     test_df.drop('Current_pincode_ID', axis=1), 4)
# make_prediction("xgb1", pred_xgb1, ts_unique_ids, sub)
# np.save('submission/stacking/oof_xgb1.npy',oof_xgb1)
# del oof_xgb1, pred_xgb1
gc.collect()


# In[14]:


# model_cat1, imp_cat1, oof_cat1, pred_cat1 = run_cv_cat(train_df, target, test_df, 4)
# make_prediction("cat1", pred_cat1, ts_unique_ids, sub)
# np.save('submission/stacking/oof_cat1.npy',oof_cat1)
# del oof_cat1, pred_cat1
# gc.collect()


# In[58]:


# train_df1 = pd.concat([train_df, target], axis=1)
# train_df1.to_csv('store_data/new_feat/train_df_feat.csv', index=None)
# test_df.to_csv('store_data/new_feat/test_df_feat.csv', index=None)
# del train_df1
gc.collect()


# In[15]:


imp_lgb1.imp = imp_lgb1.imp.astype('float')
imp_lgb1 = imp_lgb1.groupby(['feature'])['imp'].mean()
imp_lgb1 = pd.DataFrame(data=[imp_lgb1.index, imp_lgb1.values]).T
imp_lgb1.columns=['feature','imp']
imp_lgb1 = imp_lgb1.sort_values(by='imp')
plt.figure(figsize=(12,15))
plt.barh(imp_lgb1.feature, imp_lgb1.imp)


# In[16]:


imp_xgb1.imp = imp_xgb1.imp.astype('float')
imp_xgb1 = imp_xgb1.groupby(['feature'])['imp'].mean()
imp_xgb1 = pd.DataFrame(data=[imp_xgb1.index, imp_xgb1.values]).T
imp_xgb1.columns=['feature','imp']
imp_xgb1 = imp_xgb1.sort_values(by='imp')
plt.figure(figsize=(12,15))
plt.barh(imp_xgb1.feature, imp_xgb1.imp)


# In[189]:


# imp_cat1.imp = imp_cat1.imp.astype('float')
# imp_cat1 = imp_cat1.groupby(['feature'])['imp'].mean()
# imp_cat1 = pd.DataFrame(data=[imp_cat1.index, imp_cat1.values]).T
# imp_cat1.columns=['feature','imp']
# imp_cat1 = imp_cat1.sort_values(by='imp')
# plt.figure(figsize=(12,15))
# plt.barh(imp_cat1.feature, imp_cat1.imp)


# In[197]:


imp_lgb1['feature'][-25:].values


# In[17]:


inter_cols = ['disbursed_amount','asset_cost','supplied_id', 'age(in year)',
     'credit_hist_total_month', 'loan_tenure_total_month','PRI.CURRENT.BALANCE',
     'PRI.DISBURSED.AMOUNT','Employement.Type','manufacturer_id'
              
     'Bureau_desc', 'bureau_score', 'State_ID', 'stat_skew','stat_median','ltv'
     'stat_std', 'stat_mean', 'branch_date_disbursed_mean','PRI.SANCTIONED.AMOUNT',
     'manufac_date_disbursed_mean','Current_pincode_ID', 'pay_by_yourself',
     'no_of_acc_overdue','NO.OF_INQUIRIES', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']

drop_cols = ['no_of_acc','new_acc_past_month','Aadhar_flag',
    'PAN_flag','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.SANCTIONED.AMOUNT',
    'SEC.DISBURSED.AMOUNT','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.INSTAL.AMT',
    'Passport_flag','VoterID_flag']


# In[132]:


print("lgb    : ", roc_auc_score(target, oof_lgb1))
print("xgb    : ", roc_auc_score(target, oof_xgb1))
print("cat    : ", roc_auc_score(target, oof_cat1))
print("combine: ", roc_auc_score(target, (oof_lgb1 + oof_xgb1)/2))


# In[18]:


complete_df = pd.concat([train_df, test_df], axis=0)
print(complete_df.shape)

int_cols = []
for col in complete_df.columns:
    n_unique = complete_df[col].unique()
    if len(n_unique) < 300:
        print(col ," ==" ,n_unique)
    else:
        print("===", col)
        int_cols.append(col)


# In[19]:


int_cols.append('manufac_week_disbursed_mean')
int_cols.append('state_week_disbursed_mean')
int_cols.append('branch_disbursed_mean')
int_cols.append('manufac_disbursedDate_mean')
int_cols.append('state_disbursedDate_mean')

cat_cols = list(set(complete_df.columns) - set(int_cols))

len(int_cols), len(cat_cols)


# In[20]:


del complete_df
gc.collect()


# In[21]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression

def get_standard_data(train_df, test_df, which_stdc):
    """
    Args:
        train_df, test_df
        which_stdc: which one to use ['stdc_selected', 'stdc_all']
    Return:
        standarization scaled version for numeric data
        min-max scaled version for categorical data
    """
    complete_df = pd.concat([train_df, test_df], axis=0)

    stdc = StandardScaler()
    min_max = MinMaxScaler()

    if which_stdc == 'stdc_selected':
        X1 = stdc.fit_transform(complete_df[int_cols])
        X2 = min_max.fit_transform(complete_df[cat_cols])

        X1 = pd.DataFrame(data=X1, columns=int_cols)
        X2 = pd.DataFrame(data=X2, columns=cat_cols)

        complete_df = pd.concat([X1, X2], axis=1)
        del X1, X2
    elif which_stdc == 'stdc_all':
        X1 = stdc.fit_transform(complete_df)
        complete_df = pd.DataFrame(data=X1, columns=list(complete_df.columns))
        del X1
    else:
        print("pls verify flag, it should be one of ['stdc_selected', 'stdc_all']")
        
    train_df_new = complete_df.iloc[:train_df.shape[0],:]
    test_df_new  = complete_df.iloc[train_df.shape[0]:,:]
#     print(train_df_new.shape, test_df_new.shape)
    
    test_df_new = test_df_new.reset_index(drop=True)
    
    del complete_df
    gc.collect()
    return train_df_new, test_df_new
    
# tp, ts = get_standard_data(train_df, test_df)

# def get_standard_data(train_df, test_df):
#     """
#     Args:
#         train_df, test_df
#     Return:
#         standarization scaled version for numeric data
#         min-max scaled version for categorical data
#     """
#     complete_df = pd.concat([train_df, test_df], axis=0)
#     stdc = StandardScaler()
#     X1 = stdc.fit_transform(complete_df)
#     complete_df = pd.DataFrame(data=X1, columns=list(complete_df.columns))
    
#     train_df_new = complete_df.iloc[:train_df.shape[0],:]
#     test_df_new  = complete_df.iloc[train_df.shape[0]:,:]
# #     print(train_df_new.shape, test_df_new.shape)
    
#     test_df_new = test_df_new.reset_index(drop=True)
    
#     del X1, complete_df
#     gc.collect()
#     return train_df_new, test_df_new


# In[32]:



# ridge = RidgeClassifier(class_weight='balanced', alpha=10)
# ridge.fit(complete_df.iloc[:200000,:],target.iloc[:200000])

# print(roc_auc_score(target.iloc[200000:], 1+ridge.decision_function(complete_df.iloc[200000:233154,:])))
# print(ridge.score(complete_df.iloc[200000:233154,:],target.iloc[200000:]))


# In[41]:


# logistic = LogisticRegression(class_weight='balanced', n_jobs=-1, C=.1)
# logistic.fit(complete_df.iloc[:200000,:],target.iloc[:200000])

# print(roc_auc_score(target.iloc[200000:], logistic.predict_proba(complete_df.iloc[200000:233154,:])[:,1]))
# # print(roc_auc_score(target.iloc[200000:], 1+logistic.decision_function(complete_df.iloc[200000:233154,:])))
# print(logistic.score(complete_df.iloc[200000:233154,:],target.iloc[200000:]))


# In[95]:


# logistic = LogisticRegression(class_weight='balanced', n_jobs=-1, penalty='l1')
# logistic.fit(complete_df.iloc[:200000,:],target.iloc[:200000])

# print(roc_auc_score(target.iloc[200000:], logistic.predict_proba(complete_df.iloc[200000:233154,:])[:,1]))
# # print(roc_auc_score(target.iloc[200000:], 1+logistic.decision_function(complete_df.iloc[200000:233154,:])))
# print(logistic.score(complete_df.iloc[200000:233154,:],target.iloc[200000:]))


# In[57]:


def train_ridge_model(X_train, y_train, X_valid, y_valid, X_test, alpha=None):
    """
    Args:
        alpha: regularization Strength
    Return:
        oof-pred, test_preds, model, model_imp
    """
    if alpha is None: alpha = 1
    
    clf = RidgeClassifier(class_weight='balanced', alpha=alpha, random_state=1234)
    clf.fit(X_train, y_train)
    oof = 1+clf.decision_function(X_valid)
    test_pred = clf.decision_function(X_test)

    return oof, test_pred, clf

def train_logistic_model(X_train, y_train, X_valid, y_valid, X_test, penality, C=None):
    """
    Args:
        penality: 'l1' or 'l2'
        C: regularization Strength
    Return:
        oof-pred, test_preds, model, model_imp
    """
    if C is None: C = 1
    
    if penality == 'l1':
        clf = LogisticRegression(class_weight='balanced', n_jobs=-1, verbose=1,
                                 C=C, penality='l1', random_state=1234)
    elif penality == 'l2':
        clf = LogisticRegression(class_weight='balanced', n_jobs=-1, C=C,
                                 verbose=True, random_state=1234)
    else:
        print("mismatch parameter: ")
        pass
    
    clf.fit(X_train, y_train)
    oof = clf.predict_proba(X_valid)[:,1]
    test_pred = clf.predict_proba(X_test)[:,1]

    return oof, test_pred, clf



def run_cv_linear(train_df, target, test_df, model_name, which_std,
                  penality=None, alpha=None, log_C=None):
    """
    Args:
        model_name: 'logistic', 'ridge' #LogisticRegression', 'RidgeClassifier'
        penality: 'l1' or 'l2' for logistic 
        alpha: regularization for ridge
        log_C: regularization for logistic
        which_std: flag to run standardization ['stdc_selected','stdc_all']
    """
    train_df_stdc, test_df_stdc = get_standard_data(train_df, test_df, which_std)
    
    reduce_mem_usage_wo_print(train_df_stdc)
    reduce_mem_usage_wo_print(test_df_stdc)
    ##########################
    n_splits = 3
    random_seed = 1234
    
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_linear = np.zeros(len(train_df))
    predictions = np.zeros((len(test_df),n_splits))
    clfs = []
##########################
    for fold_, (train_index, valid_index) in enumerate(folds.split(train_df, target)):
        print(train_index.shape, valid_index.shape)
        print("Fold {}".format(fold_))
    
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        X_train, X_valid = train_df_stdc.iloc[train_index,:], train_df_stdc.iloc[valid_index,:]
                
        if log_C is None: log_C = 1
        if alpha is None: alpha = 1
        if penality is None: penality = 'l2'
            
        if model_name == 'logistic':
            oof, test_pred, clf = train_logistic_model(X_train, y_train, 
                                                       X_valid, y_valid, 
                                                       test_df_stdc, penality, C=log_C)
        elif model_name == 'ridge':
            oof, test_pred, clf = train_ridge_model(X_train, y_train, 
                                                    X_valid, y_valid, 
                                                    test_df_stdc, alpha=alpha)
        else:
            print("Incorrect model name")
            pass

    
        oof_linear[valid_index] = oof
        predictions[:,fold_] = test_pred
        
        clfs.append(clf)
        
        score = roc_auc_score(y_valid, oof)
        print( "  auc = ", score )
        print("="*60)
    
    return clfs, oof_linear, predictions


# In[23]:


model_logistic, oof_logistic, pred_logistic = run_cv_linear(train_df, target, 
                                                            test_df,'logistic',
                                                            'stdc_selected'
                                                           )
make_prediction("logistic1", pred_logistic, ts_unique_ids, sub)
np.save('submission/stacking/oof_logistic1.npy',oof_logistic)
del oof_logistic, pred_logistic
gc.collect()

# make_prediction("cat_stratified", pred_cat1, ts_unique_ids, sub)


# In[24]:


model_ridge, oof_ridge, pred_ridge = run_cv_linear(train_df, target, 
                                                   test_df,'ridge', 
                                                   'stdc_selected')
make_prediction("ridge1", pred_ridge, ts_unique_ids, sub)
np.save('submission/stacking/oof_ridge1.npy',oof_ridge)
del oof_ridge, pred_ridge
gc.collect()


# In[26]:


# try:
#     make_prediction("ridge1", pred_ridge, ts_unique_ids, sub)
#     np.save('submission/stacking/oof_ridge1.npy',oof_ridge)
#     del oof_ridge, pred_ridge
#     gc.collect()
# except:
#     print("could not save ridge")


# In[35]:


drop_cols = ['no_of_acc',
             'Aadhar_flag',
#              'VoterID_flag',
#              'SEC.NO.OF.ACCTS',
#              'SEC.ACTIVE.ACCTS',
             'SEC.SANCTIONED.AMOUNT',
             'SEC.DISBURSED.AMOUNT',
#              'SEC.OVERDUE.ACCTS',
             'SEC.CURRENT.BALANCE',
#              'SEC.INSTAL.AMT'
            ]

train_df.drop(drop_cols, axis=1, inplace=True)
test_df.drop(drop_cols, axis=1, inplace=True)


# In[105]:


import h2o
h2o.init()
# h2o.shutdown(prompt=False)


# In[106]:



# train = h2o.H2OFrame(pd.concat([feature_df.iloc[:target.shape[0]], target], axis=1))                     
# test  = h2o.H2OFrame(feature_df.iloc[target.shape[0]:].reset_index(drop=True))

train = h2o.H2OFrame(pd.concat([train_df, target], axis=1))                     
test  = h2o.H2OFrame(test_df)
print(train.shape, test.shape)

gc.collect()

x = train.columns
y = "loan_default"
x.remove(y)

train[y] = train[y].asfactor()
# test[y]  = test[y].asfactor()

# aml = H2OAutoML(max_models=20, seed=1337, max_runtime_secs=14400, nfolds=3)
# aml.train(x = x, y = y,
#           training_frame = train,
#           leaderboard_frame = test)
# lb = aml.leaderboard
# lb.head(rows=lb.nrows)

gc.collect()


# In[108]:


# del train, test
gc.collect()
# train.head()


# In[109]:


from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
# from h2o.estimators.kmeans import H2OKMeansEstimator
# from h2o.estimators.xgboost import H2OXGBoostEstimator


# In[110]:


def make_h20_prediction(file_path, df, test_ids, sub_df):
    """
    Args:
        file_path: file-name with base-path as "submission"
        df: H2o data-frame with 3 columns [predict, p0, p1]
        test_ids: test_ids
        sub_df: submission data-frame
        
    Return:
        output a file with given name
        
    Example: 
    >>> make_prediction(file_path, predictions, ts_unique_ids, sub)
    """
    sub_df = pd.DataFrame({"ID_code":test_ids})
    sub_df["target"] = df
    sub_df.columns = sub.columns

    sub_df.to_csv('submission/stacking/{}.csv'.format(file_path), index=None)
    print("successfully saved")


# In[87]:


# drf_model_deep = H2ORandomForestEstimator( nfolds=3, seed=1234,
#                                            keep_cross_validation_predictions=True,
#                                            fold_assignment = 'stratified',
# #                                            histogram_type = 'QuantilesGlobal',
# #                                            categorical_encoding = 'eigen',
#                                            stopping_metric = 'auc',
#                                            ntrees = 50,
#                                            balance_classes = True
#                                            )
# drf_model_deep.train(x, y, training_frame=train)#, validation_frame=test)
# print(drf_model_deep.accuracy())


# In[ ]:


# drf_model_deep = H2ORandomForestEstimator( nfolds=3, seed=1234,
#                                            keep_cross_validation_predictions=True,
#                                            fold_assignment = 'stratified',
# #                                            histogram_type = 'QuantilesGlobal',
# #                                            categorical_encoding = 'eigen',
#                                            stopping_metric = 'auc',
#                                            ntrees = 50,
#                                            balance_classes = True,
#                                            categorical_encoding=''
#                                            )
# drf_model_deep.train(x, y, training_frame=train)#, validation_frame=test)
# print(drf_model_deep.accuracy())


# In[ ]:


# drf_model_deep = H2ORandomForestEstimator( nfolds=3, seed=1234,
#                                            keep_cross_validation_predictions=True,
#                                            fold_assignment = 'stratified',
# #                                            histogram_type = 'QuantilesGlobal',
# #                                            categorical_encoding = 'eigen',
#                                            stopping_metric = 'auc',
#                                            ntrees = 50,
#                                            balance_classes = True,
#                                            categorical_encoding=''
#                                            )
# drf_model_deep.train(x, y, training_frame=train)#, validation_frame=test)
# print(drf_model_deep.accuracy())


# In[94]:


# gbm_model_deep = H2OGradientBoostingEstimator(  nfolds=2, seed=1234,
#                                                 fold_assignment = 'stratified',
# #                                                 score_validation_sampling='stratified',
#                                                 ntrees = 100,
#                                                 learn_rate = 0.05,
#                                                 max_depth = 3,
#                                                 stopping_rounds = 5, 
#                                                 stopping_tolerance = 1e-3,
#                                                 stopping_metric = "AUC",
#                                                 categorical_encoding='sort_by_response',
#                                                 sample_rate = 0.8,
#                                                 col_sample_rate = 0.7,
#                                                 keep_cross_validation_predictions=True
#                                              )
# gbm_model_deep.train(x, y, training_frame=train)#, validation_frame=test)
# print(gbm_model_deep.accuracy())



# In[ ]:


# gbm_model_deep = H2OGradientBoostingEstimator(  nfolds=2, seed=1234,
#                                                 fold_assignment = 'stratified',
# #                                                 score_validation_sampling='stratified',
#                                                 ntrees = 100,
#                                                 learn_rate = 0.05,
#                                                 max_depth = 4,
#                                                 stopping_rounds = 5, 
#                                                 stopping_tolerance = 1e-3,
#                                                 stopping_metric = "AUC",
#                                                 sample_rate = 0.8,
#                                                 categorical_encoding='enum',
#                                                 col_sample_rate = 0.7,
#                                                 keep_cross_validation_predictions=True
#                                              )
# gbm_model_deep.train(x, y, training_frame=train)#, validation_frame=test)
# print(gbm_model_deep.accuracy())


# In[111]:



dl_model.train(x, y, training_frame=train)#, validation_frame=test)
print(dl_model.accuracy())

model_path = h2o.save_model(model=dl_model, path="submission/stacking/", force=True)
print(model_path)
# var_df = pd.DataFrame(dl_model.varimp(), 
#         columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])
# print(var_df.shape)
# var_df.head(10)
pred = dl_model.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("dl_model_test", pred_p1, ts_unique_ids, sub)

oof = dl_model.cross_validation_holdout_predictions()
oof = oof.as_data_frame()
oof_p1 = oof['p1'].values
make_h20_prediction("dl_model_oof", oof_p1, tr_unique_ids, sub)

del pred, pred_p1, oof, oof_p1
gc.collect()

# dl_model.varimp_plot()


# In[145]:


# save the model
# model_path = h2o.save_model(model=dl_model, path="submission/stacking/", force=True)

print(model_path)
# /tmp/mymodel/DeepLearning_model_python_1441838096933

# load the model
# saved_model = h2o.load_model(model_path)


# In[113]:




dl_model_deep.train(x, y, training_frame=train)#, validation_frame=test)
print(dl_model_deep.accuracy())

model_path = h2o.save_model(model=dl_model_deep, path="submission/stacking/", force=True)
print(model_path)

pred = dl_model_deep.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("dl_model_deep_test", pred_p1, ts_unique_ids, sub)

oof = dl_model_deep.cross_validation_holdout_predictions()
oof = oof.as_data_frame()
oof_p1 = oof['p1'].values
make_h20_prediction("dl_model_deep_oof", oof_p1, tr_unique_ids, sub)

# del dl_model_deep, pred, pred_p1, oof, oof_p1
gc.collect()

# dl_model_deep.varimp_plot()


# In[73]:


len(set(train.columns).intersection(test.columns)), len(set(train.columns))


# In[74]:


# test.columns


# In[114]:


gbm_model = H2OGradientBoostingEstimator(nfolds=3, seed=1234,
                                         balance_classes = True,
                                         col_sample_rate = 0.7,
                                         learn_rate=0.1,
#                                          nbins = 128,
                                         fold_assignment='stratified',
                                         stopping_rounds=25,
                                         categorical_encoding='enum',
                                         keep_cross_validation_predictions=True
                                        )
gbm_model.train(x, y, training_frame=train)#, validation_frame=test)
print(gbm_model.accuracy())

# var_df = pd.DataFrame(gbm_model.varimp(), 
#         columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])

model_path = h2o.save_model(model=gbm_model, path="submission/stacking/", force=True)
print(model_path)

pred = gbm_model.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("gbm_model_test", pred_p1, ts_unique_ids, sub)

oof = gbm_model.cross_validation_holdout_predictions()
oof = oof.as_data_frame()
oof_p1 = oof['p1'].values
make_h20_prediction("gbm_model_oof", oof_p1, tr_unique_ids, sub)

# del gbm_model, pred, pred_p1, oof, oof_p1
gc.collect()


# gbm_model.varimp_plot()


# In[ ]:



gbm_model_one_hot.train(x, y, training_frame=train)#, validation_frame=test)
try:
    print(gbm_model_one_hot.accuracy())
except:
    pass
# var_df = pd.DataFrame(gbm_model_one_hot.varimp(), 
#         columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])

model_path = h2o.save_model(model=gbm_model_one_hot, path="submission/stacking/", force=True)
print(model_path)

pred = gbm_model_one_hot.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("gbm_model_one_hot_test", pred_p1, ts_unique_ids, sub)

oof = gbm_model_one_hot.cross_validation_holdout_predictions()
oof = oof.as_data_frame()
oof_p1 = oof['p1'].values
make_h20_prediction("gbm_model_one_hot_oof", oof_p1, tr_unique_ids, sub)

del gbm_model_one_hot, pred, pred_p1, oof, oof_p1
gc.collect()


# gbm_model_one_hot.varimp_plot()


# In[115]:


gbm_model_sort.train(x, y, training_frame=train)#, validation_frame=test)
try:
    print(gbm_model_sort.accuracy())
except:
    pass
# var_df = pd.DataFrame(gbm_model_sort.varimp(), 
#         columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])

model_path = h2o.save_model(model=gbm_model_sort, path="submission/stacking/", force=True)
print(model_path)

pred = gbm_model_sort.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("gbm_model_sort_test", pred_p1, ts_unique_ids, sub)

oof = gbm_model_sort.cross_validation_holdout_predictions()
oof = oof.as_data_frame()
oof_p1 = oof['p1'].values
make_h20_prediction("gbm_model_sort_oof", oof_p1, tr_unique_ids, sub)

# del gbm_model_sort, pred, pred_p1, oof, oof_p1
gc.collect()


# gbm_model_sort.varimp_plot()


# In[ ]:


gbm_model_deep = H2OGradientBoostingEstimator(  nfolds=3, seed=1234,
                                                fold_assignment = 'stratified',
#                                                 score_validation_sampling='stratified',
                                                ntrees = 10000,
                                                learn_rate = 0.1,
                                                max_depth = 4,
                                                stopping_rounds = 5, 
                                                stopping_tolerance = 1e-3,
                                                stopping_metric = "AUC",
                                                sample_rate = 0.8,
                                                col_sample_rate = 0.7,
                                                keep_cross_validation_predictions=True
                                             )

gbm_model_sort = H2OGradientBoostingEstimator(nfolds=3, seed=1234,
                                         balance_classes = True,
                                         col_sample_rate = 0.7,
                                         learn_rate=0.1,
                                         ntrees = 10000,
#                                          nbins = 128,sort_internal
# sort_by_response
                                         categorical_encoding='sortbyresponse',
                                         fold_assignment='stratified',
                                         stopping_rounds=25,
#                                          categorical_encoding='enum',
                                         keep_cross_validation_predictions=True
                                        )


gbm_model_deep.train(x, y, training_frame=train)#, validation_frame=test)
try:
    print(gbm_model_deep.accuracy())
except:
    pass
# var_df = pd.DataFrame(gbm_model.varimp(), 
#         columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])

model_path = h2o.save_model(model=gbm_model_deep, path="submission/stacking/", force=True)
print(model_path)

pred = gbm_model_deep.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("gbm_model_deep_test", pred_p1, ts_unique_ids, sub)

oof = gbm_model_deep.cross_validation_holdout_predictions()
oof = oof.as_data_frame()
oof_p1 = oof['p1'].values
make_h20_prediction("gbm_model_deep_oof", oof_p1, tr_unique_ids, sub)

del gbm_model_deep, pred, pred_p1, oof, oof_p1
gc.collect()



# gbm_model_deep.varimp_plot()

                                                             


# In[ ]:


# model = H2OGradientBoostingEstimator(distribution='bernoulli',
#                                         ntrees=100,
#                                         max_depth=4,
#                                         learn_rate=0.1


# In[116]:



drf_model.train(x, y, training_frame=train)#, validation_frame=test)
try:
    print(drf_model.accuracy())
except:
    pass
# var_df = pd.DataFrame(drf_model.varimp(), 
#         columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])
model_path = h2o.save_model(model=drf_model, path="submission/stacking/", force=True)
print(model_path)

pred = drf_model.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("drf_model_test", pred_p1, ts_unique_ids, sub)

oof = drf_model.cross_validation_holdout_predictions()
oof = oof.as_data_frame()
oof_p1 = oof['p1'].values
make_h20_prediction("drf_model_oof", oof_p1, tr_unique_ids, sub)

del drf_model, pred, pred_p1, oof, oof_p1
gc.collect()


# drf_model.varimp_plot()


# In[117]:


drf_model_deep = H2ORandomForestEstimator( nfolds=3, seed=1234,
                                           keep_cross_validation_predictions=True,
                                           fold_assignment = 'stratified',
#                                            histogram_type = 'QuantilesGlobal',
#                                            categorical_encoding = 'eigen',
                                           stopping_metric = 'auc',
                                           ntrees = 200,
                                           balance_classes = True
                                           )
drf_model_deep.train(x, y, training_frame=train)#, validation_frame=test)
print(drf_model_deep.accuracy())

model_path = h2o.save_model(model=drf_model_deep, path="submission/stacking/", force=True)
print(model_path)

pred = drf_model_deep.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("drf_model_deep_test", pred_p1, ts_unique_ids, sub)

oof = drf_model_deep.cross_validation_holdout_predictions()
oof = oof.as_data_frame()
oof_p1 = oof['p1'].values
make_h20_prediction("drf_model_deep_oof", oof_p1, tr_unique_ids, sub)

# del drf_model_deep, pred, pred_p1, oof, oof_p1
gc.collect()



# drf_model_deep.varimp_plot()


# In[47]:





# In[90]:


# drf_model_sort = H2ORandomForestEstimator( nfolds=3, seed=1234,
#                                            keep_cross_validation_predictions=True,
#                                            fold_assignment = 'stratified',
# #                                            histogram_type = 'QuantilesGlobal',
#                                            categorical_encoding = 'sort_by_response',
#                                            stopping_metric = 'auc',
# #                                            ntrees = 10,
#                                            min_rows=5,
#                                            balance_classes = True
#                                            )
# drf_model_sort.train(x, y, training_frame=train)#, validation_frame=test)
# print(drf_model_sort.accuracy())

# model_path = h2o.save_model(model=drf_model_sort, path="submission/stacking/", force=True)
# print(model_path)

# pred = drf_model_sort.predict(test)
# pred = pred.as_data_frame()
# pred_p1 = pred['p1'].values
# make_h20_prediction("drf_model_sort_test", pred_p1, ts_unique_ids, sub)

# oof = drf_model_sort.cross_validation_holdout_predictions()
# oof = oof.as_data_frame()
# oof_p1 = oof['p1'].values
# make_h20_prediction("drf_model_sort_oof", oof_p1, tr_unique_ids, sub)

# # del drf_model_sort, pred, pred_p1, oof, oof_p1
# gc.collect()



# # drf_model_sort.varimp_plot()


# In[92]:


# drf_model_binary = H2ORandomForestEstimator( nfolds=3, seed=1234,
#                                            keep_cross_validation_predictions=True,
#                                            fold_assignment = 'stratified',
# #                                            histogram_type = 'QuantilesGlobal',
#                                            categorical_encoding = 'binary',
#                                            stopping_metric = 'auc',
#                                            ntrees = 100,
#                                            min_rows=5,
#                                            balance_classes = True
#                                            )
# drf_model_binary.train(x, y, training_frame=train)#, validation_frame=test)
# print(drf_model_binary.accuracy())

# model_path = h2o.save_model(model=drf_model_binary, path="submission/stacking/", force=True)
# print(model_path)

# pred = drf_model_binary.predict(test)
# pred = pred.as_data_frame()
# pred_p1 = pred['p1'].values
# make_h20_prediction("drf_model_binary_test", pred_p1, ts_unique_ids, sub)

# oof = drf_model_binary.cross_validation_holdout_predictions()
# oof = oof.as_data_frame()
# oof_p1 = oof['p1'].values
# make_h20_prediction("drf_model_binary_oof", oof_p1, tr_unique_ids, sub)

# # del drf_model_binary, pred, pred_p1, oof, oof_p1
# gc.collect()



# # drf_model_binary.varimp_plot()


# In[ ]:



# drf_model = H2ORandomForestEstimator()
# drf_model.train(x, y, training_frame=train, validation_frame=test)
# # var_df = pd.DataFrame(drf_model.varimp(), 
# #         columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])
# drf_model.varimp_plot()


# In[97]:


# glm_model = H2OGeneralizedLinearEstimator(nfolds=3, seed=1234,
#                                           keep_cross_validation_predictions=True,
#                                           fold_assignment = 'stratified',
# #                                           interactions=x,
#                                           family="binomial")
# glm_model.train(x, y, training_frame=train)#, validation_frame=test)
# try:
#     print(glm_model.accuracy())
# except:
#     pass
# model_path = h2o.save_model(model=glm_model, path="submission/stacking/", force=True)
# print(model_path)

# pred = glm_model.predict(test)
# pred = pred.as_data_frame()
# pred_p1 = pred['p1'].values
# make_h20_prediction("glm_model_test", pred_p1, ts_unique_ids, sub)

# oof = glm_model.cross_validation_holdout_predictions()
# oof = oof.as_data_frame()
# oof_p1 = oof['p1'].values
# make_h20_prediction("glm_model_oof", oof_p1, tr_unique_ids, sub)

# del glm_model, pred, pred_p1, oof, oof_p1
# gc.collect()

# # glm_model.std_coef_plot()


# In[49]:



# glm_model_bal = H2OGeneralizedLinearEstimator(nfolds=3, seed=1234,
#                                               keep_cross_validation_predictions=True,
#                                               fold_assignment = 'stratified',
#                                               family="binomial", 
#                                               lambda_search=True,
#                                               balance_classes=True)
# glm_model_bal.train(x, y, training_frame=train)#, validation_frame=test)
# try:
#     print(glm_model_bal.accuracy())
# except:
#     pass

# model_path = h2o.save_model(model=glm_model_bal, path="submission/stacking/", force=True)
# print(model_path)

# pred = glm_model_bal.predict(test)
# pred = pred.as_data_frame()
# pred_p1 = pred['p1'].values
# make_h20_prediction("glm_model_bal_test", pred_p1, ts_unique_ids, sub)

# oof = glm_model_bal.cross_validation_holdout_predictions()
# oof = oof.as_data_frame()
# oof_p1 = oof['p1'].values
# make_h20_prediction("glm_model_bal_oof", oof_p1, tr_unique_ids, sub)

# del glm_model_bal, pred, pred_p1, oof, oof_p1
# gc.collect()


# # glm_model_bal.std_coef_plot()


# In[118]:


inter_cols = ['ltv','disbursed_amount','asset_cost','supplied_id', 'age(in year)',
     'credit_hist_total_month', 'loan_tenure_total_month','PRI.CURRENT.BALANCE',
     'PRI.DISBURSED.AMOUNT','PRI.SANCTIONED.AMOUNT','Bureau_desc']

drop_cols = ['no_of_acc', 'NO.OF_INQUIRIES','new_acc_past_month',
    'no_of_acc_overdue','loan_tenure_year','disbursal_month','Aadhar_flag',
    'PAN_flag','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.SANCTIONED.AMOUNT',
    'SEC.DISBURSED.AMOUNT','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.INSTAL.AMT',
    'Passport_flag','VoterID_flag','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']

glm_model_inter = H2OGeneralizedLinearEstimator(nfolds=3, seed=1234,
                                                keep_cross_validation_predictions=True,
                                                fold_assignment = 'stratified',
                                                ignored_columns = drop_cols,
#                                                 interactions = inter_cols,
                                                family="binomial", 
                                                lambda_search=True,
                                                balance_classes=True,
                                                remove_collinear_columns = True)
glm_model_inter.train(x, y, training_frame=train)#, validation_frame=test)

print(glm_model_inter.accuracy())

model_path = h2o.save_model(model=glm_model_inter, path="submission/stacking/", force=True)
print(model_path)

pred = glm_model_inter.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("glm_model_inter_test", pred_p1, ts_unique_ids, sub)

oof = glm_model_inter.cross_validation_holdout_predictions()
oof = oof.as_data_frame()
oof_p1 = oof['p1'].values
make_h20_prediction("glm_model_inter_oof", oof_p1, tr_unique_ids, sub)

# del glm_model_inter, pred, pred_p1, oof, oof_p1
gc.collect()



# glm_model_inter.std_coef_plot()


# In[124]:


# from h2o.estimators.isolation_forest import H2OIsolationForestEstimator

# ife_model = H2OIsolationForestEstimator(nfolds=3, seed=1234,
#                                                 keep_cross_validation_predictions=True,
#                                                 fold_assignment = 'stratified',
# #                                                 ignored_columns = drop_cols,
# # #                                                 interactions = inter_cols,
# #                                                 family="binomial", 
# #                                                 lambda_search=True,
# #                                                 balance_classes=True,
# #                                                 remove_collinear_columns = True
#                                              )
# ife_model.train(x, y, training_frame=train)#, validation_frame=test)
# print(ife_model.accuracy())

all_models = [dl_model, dl_model_deep, gbm_model, gbm_model_sort,
              drf_model_deep, glm_model_inter]
for model in all_models:
    print(model.model_id, model.accuracy())
del all_models
gc.collect()


# In[131]:


# from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

# stack_model = H2OStackedEnsembleEstimator(#model_id="my_ensemble", 
#                                     training_frame=train, 
#                                     base_models=[dl_model, dl_model_deep, 
#                                                  gbm_model, gbm_model_sort,
#                                                  drf_model_deep, glm_model_inter],
#                                     seed=1234,
                                          
                                         
#                                          )#.model_id, drf_model.model_id])

# stack_model.train(x=x, y=y, training_frame=train)#, validation_frame=test)
# stack_model.accuracy()


# In[122]:


# stack_model.accuracy(), gbm_model_deep.accuracy()


# In[99]:


drf_model_deep.accuracy()


# In[ ]:


# h2o.shutdown(prompt=False)


# In[165]:


all_models = [dl_model, dl_model_deep, gbm_model, gbm_model_sort,
              drf_model_deep, glm_model_inter]
all_preds = []
for model in all_models:
    print(model.model_id, model.accuracy())
    all_preds.append(model.cross_validation_holdout_predictions().as_data_frame()['p1'])
# all_preds = np.array(all_preds)
# all_preds.shape


# In[166]:


oof_lgb1 = np.load('submission/stacking/oof_lgb11.npy')
all_preds.append(oof_lgb1)
all_preds = np.array(all_preds)
all_preds.shape


# In[167]:


all_preds = pd.DataFrame(data=all_preds.T, columns = ['dl','dld','gbm',
                                                      'gbmd','drf','glm','lgb'])
print(all_preds.head())


oofs = all_preds.copy()
corr = oofs.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

roc_score = {}
for col in oofs.columns:
    roc_score[col] = roc_auc_score(target, oofs[col])
#     print(col.ljust(8), "==>", roc_auc_score(y_valid,preds[col]))
roc_score = pd.DataFrame(data=[roc_score.keys(), roc_score.values()]).T
roc_score.columns = ['model','roc']
roc_score['roc_norm'] = (roc_score.roc - roc_score.roc.min())/                         (roc_score.roc.max() - roc_score.roc.min())

plt.figure(figsize=(16,4))
g = sns.barplot(x='model',y='roc_norm',data=roc_score)

for index, row in roc_score.iterrows():
    g.text(row.name,row.roc_norm, round(row.roc,3), color='black', ha="center")

    
#     # print(roc_auc_score(target, oofs1.mean(1)))
# oofs2 = oofs1.rank()
# oofs2 = oofs2/oofs2.max()
# oofs2.max()


# In[191]:


oof_xgb1 = np.load('submission/stacking/saturday/oof_xgb1.npy')
oof_xgb2 = np.load('submission/stacking/saturday/oof_xgb_inter.npy')
oof_xgb3 = np.load('submission/stacking/saturday/oof_xgb_inter_3way1.npy')
oof_xgb4 = np.load('submission/stacking/saturday/oof_xgb_inter_3way2.npy')

oof_lgb1 = np.load('submission/stacking/saturday/oof_lgb1.npy')
oof_lgb2 = np.load('submission/stacking/saturday/oof_lgb_inter.npy')
oof_lgb3 = np.load('submission/stacking/saturday/oof_lgb_inter_3way1.npy')

oof_ridge = np.load('submission/stacking/saturday/oof_ridge1.npy')
oof_logistic = np.load('submission/stacking/saturday/oof_logistic1.npy')

oof_dl = pd.read_csv('submission/stacking/saturday/dl_model_oof.csv')
oof_dl_deep = pd.read_csv('submission/stacking/saturday/dl_model_deep_oof.csv')

oof_drf = pd.read_csv('submission/stacking/saturday/drf_model_oof.csv')
oof_drf_deep = pd.read_csv('submission/stacking/saturday/drf_model_deep_oof.csv')

oof_gbm = pd.read_csv('submission/stacking/saturday/gbm_model_oof.csv')
oof_gbm_deep = pd.read_csv('submission/stacking/saturday/gbm_model_deep_oof.csv')

oof_glm = pd.read_csv('submission/stacking/saturday/glm_model_oof.csv')
oof_glm_bal = pd.read_csv('submission/stacking/saturday/glm_model_inter_oof.csv')
oof_glm_inter = pd.read_csv('submission/stacking/saturday/glm_model_bal_oof.csv')


oofs_purane = np.column_stack([oof_xgb1,oof_xgb2, oof_xgb3, oof_xgb4,
                 oof_lgb1, oof_lgb2, oof_lgb3, oof_ridge, oof_logistic,
                 oof_dl.loan_default.values, oof_dl_deep.loan_default.values,
                 oof_drf.loan_default.values, oof_drf_deep.loan_default.values,
                 oof_gbm.loan_default.values, oof_gbm_deep.loan_default.values,
                 oof_glm.loan_default.values, oof_glm_bal.loan_default.values,
                 oof_glm_inter.loan_default.values])

oofs_purane = pd.DataFrame(data=oofs_purane, columns=['xgb1', 'xgb2', 'xgb3', 'xgb4',
 'lgb1', 'lgb2', 'lgb3','ridge','logistic', 'dl','dll', 'drf','drfd','gbm','gbmd',
 'glm','glmd','glmdd'])

oofs_purane.columns = [col+'_p' for col in oofs_purane.columns]
oofs_purane.head()


# In[198]:


pred_all = pd.concat([oofs, oofs_purane], axis=1)
oofs1 = oofs.copy()
oofs = pred_all.copy()
oofs.shape, pred_all.shape


# In[199]:


corr = oofs.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

roc_score = {}
for col in oofs.columns:
    roc_score[col] = roc_auc_score(target, oofs[col])
#     print(col.ljust(8), "==>", roc_auc_score(y_valid,preds[col]))
roc_score = pd.DataFrame(data=[roc_score.keys(), roc_score.values()]).T
roc_score.columns = ['model','roc']
roc_score['roc_norm'] = (roc_score.roc - roc_score.roc.min())/                         (roc_score.roc.max() - roc_score.roc.min())

plt.figure(figsize=(16,4))
g = sns.barplot(x='model',y='roc_norm',data=roc_score)

for index, row in roc_score.iterrows():
    g.text(row.name,row.roc_norm, round(row.roc,3), color='black', ha="center")


# In[151]:


print(roc_auc_score(target, oofs.mean(axis=1)))
oofs1 = oofs.rank()
oofs1 = oofs1/oofs1.max()
print(roc_auc_score(target, oofs1.mean(axis=1)))


# In[152]:


nested_list = [['a','b',1],['c','d',3],['a','f',8],['a','c',5]]
from collections import defaultdict
from itertools import combinations

d = defaultdict(list)    

for i, j, _ in nested_list:
    d[i].append(j)
     

{k: list(combinations(v, 2)) for k, v in d.items()}
# {'a': [('b', 'f'), ('b', 'c'), ('f', 'c')], 'c': []}


# In[205]:


# a = ['a','b','c','d']
# for cur_comb in [list(combinations(a,2)) + list(combinations(a,3))]:
#     print(cur_comb)


# In[206]:


oofs1 = all_preds.copy()

mean_score = []
rank_score = []
count = 0
for r in range(2, oofs1.shape[1]):
    for cur_comb in list(combinations(oofs1.columns, r)):
#         print(list(cur_comb), "==>" end=" ")
#         score = roc_auc_score(target, oofs[list(cur_comb)].mean(axis=1))
#         mean_score.append(score)
# #         print("mean: ", np.round(score,4), end=" ")
#         oofs1 = oofs[list(cur_comb)].rank()
#         oofs1 = oofs1/oofs1.max()
#         score = roc_auc_score(target, oofs1.mean(axis=1))
#         rank_score.append(score)
# #         print(" rank_avg: ",np.round(score,4))
        count += 1
    print("==>", end=" ")
print("total combinations: ", count)


# In[ ]:


# mean_score = []
# rank_score = []
# count = 0
# for r in range(2, oofs.shape[1]):
#     for cur_comb in list(combinations(oofs.columns, r)):
# #         print(list(cur_comb), "==>" end=" ")
#         score = roc_auc_score(target, oofs[list(cur_comb)].mean(axis=1))
#         mean_score.append(score)
# #         print("mean: ", np.round(score,4), end=" ")
#         oofs1 = oofs[list(cur_comb)].rank()
#         oofs1 = oofs1/oofs1.max()
#         score = roc_auc_score(target, oofs1.mean(axis=1))
#         rank_score.append(score)
# #         print(" rank_avg: ",np.round(score,4))
#         count += 1
#     print("==>", end=" ")

    
def scoring(r)   
    mean_score = []
    rank_score = []
    count = 0
    for r in range(2, oofs1.shape[1]):
        for cur_comb in list(combinations(oofs1.columns, r)):
    #         print(list(cur_comb), "==>" end=" ")
            score = roc_auc_score(target, oofs1[list(cur_comb)].mean(axis=1))
            mean_score.append(score)
    #         print("mean: ", np.round(score,4), end=" ")
            oofs_ = oofs1[list(cur_comb)].rank()
            oofs_ = oofs_/oofs_.max()
            score = roc_auc_score(target, oofs_.mean(axis=1))
            rank_score.append(score)
    #         print(" rank_avg: ",np.round(score,4))
            count += 1
#         print("==>", end=" ")
    return (mean_score, rank_score)


# In[210]:


get_ipython().run_cell_magic('time', '', '\nimport multiprocessing as mp\n\ndef scoring(r):   \n    mean_score = []\n    rank_score = []\n#     for r in range(2, oofs1.shape[1]):\n    for cur_comb in list(combinations(oofs1.columns, r)):\n#         print(list(cur_comb), "==>" end=" ")\n        score = roc_auc_score(target, oofs1[list(cur_comb)].mean(axis=1))\n        mean_score.append(score)\n#         print("mean: ", np.round(score,4), end=" ")\n        oofs_ = oofs1[list(cur_comb)].rank()\n        oofs_ = oofs_/oofs_.max()\n        score = roc_auc_score(target, oofs_.mean(axis=1))\n        rank_score.append(score)\n#         print(" rank_avg: ",np.round(score,4))\n#         print("==>", end=" ")\n    return (mean_score, rank_score)\n\npool = mp.Pool(mp.cpu_count())\nresults = pool.map(scoring, [r for r in np.arange(2, oofs1.shape[1])])\npool.close()')


# In[211]:


get_ipython().run_cell_magic('time', '', '\nmean_score = []\nrank_score = []\nfor r in range(2, oofs1.shape[1]):\n    for cur_comb in list(combinations(oofs1.columns, r)):\n#         print(list(cur_comb), "==>" end=" ")\n        score = roc_auc_score(target, oofs1[list(cur_comb)].mean(axis=1))\n        mean_score.append(score)\n#         print("mean: ", np.round(score,4), end=" ")\n        oofs_ = oofs1[list(cur_comb)].rank()\n        oofs_ = oofs_/oofs_.max()\n        score = roc_auc_score(target, oofs_.mean(axis=1))\n        rank_score.append(score)')


# In[215]:


gc.collect()


# In[216]:


get_ipython().run_cell_magic('time', '', '\nimport multiprocessing as mp\n\ndef get_score(r):   \n    mean_score = []\n    rank_score = []\n#     for r in range(2, oofs1.shape[1]):\n    for cur_comb in list(combinations(oofs.columns, r)):\n#         print(list(cur_comb), "==>" end=" ")\n        score = roc_auc_score(target, oofs[list(cur_comb)].mean(axis=1))\n        mean_score.append(score)\n#         print("mean: ", np.round(score,4), end=" ")\n        oofs_ = oofs[list(cur_comb)].rank()\n        oofs_ = oofs_/oofs_.max()\n        score = roc_auc_score(target, oofs_.mean(axis=1))\n        rank_score.append(score)\n#         print(" rank_avg: ",np.round(score,4))\n#         print("==>", end=" ")\n    return (mean_score, rank_score)\n\npool = mp.Pool(mp.cpu_count())\nresults = pool.map(get_score, [r for r in np.arange(2, oofs.shape[1])])\npool.close()\ngc.collect()')


# In[221]:


gc.collect()


# In[200]:


mean_score = []
rank_score = []
count = 0
for r in range(2, oofs.shape[1]):
    for cur_comb in list(combinations(oofs.columns, r)):
#         print(list(cur_comb), "==>" end=" ")
        score = roc_auc_score(target, oofs[list(cur_comb)].mean(axis=1))
        mean_score.append(score)
#         print("mean: ", np.round(score,4), end=" ")
        oofs1 = oofs[list(cur_comb)].rank()
        oofs1 = oofs1/oofs1.max()
        score = roc_auc_score(target, oofs1.mean(axis=1))
        rank_score.append(score)
#         print(" rank_avg: ",np.round(score,4))
        count += 1
    print("==>", end=" ")
    


# In[181]:


fig, ax = plt.subplots(2,1,figsize=(18,8))
ax[0].plot(mean_score, '-o')
ax[0].set_title("mean_score")

ax[1].plot(rank_score, '-o')
ax[1].set_title("rank_score")


# In[182]:


fig, ax = plt.subplots(2,1,figsize=(18,8))
sns.distplot(mean_score, ax=ax[0])
sns.distplot(rank_score, ax=ax[1])


# In[ ]:


fig, ax = plt.subplots(2,1,figsize=(18,8))
ax[0].plot(mean_score, '-o')
ax[0].set_title("mean_score")

ax[1].plot(rank_score, '-o')
ax[1].set_title("rank_score")


# In[34]:


# train_df, test_df = get_standard_data(train_df, test_df)


# In[35]:


# model_lgb2, imp_lgb2, oof_lgb2, pred_lgb2 = run_cv_lgb(train_df, target, test_df)
# make_prediction("lgb2", pred_lgb2, ts_unique_ids, sub)
# np.save('submission/stacking/oof_lgb2.npy',oof_lgb2)
# del oof_lgb2, pred_lgb2
# gc.collect()


# In[36]:


# model_xgb2, imp_xgb2, oof_xgb2, pred_xgb2 = run_cv_xgb(train_df, target, test_df, 4)
# make_prediction("xgb2", pred_xgb2, ts_unique_ids, sub)
# np.save('submission/stacking/oof_xgb2.npy',oof_xgb2)
# del oof_xgb2, pred_xgb2
# gc.collect()


# In[37]:



# model_cat2, imp_cat2, oof_cat2, pred_cat2 = run_cv_cat(train_df, target, test_df, 4)
# make_prediction("cat2", pred_cat2, ts_unique_ids, sub)
# np.save('submission/stacking/oof_cat2.npy',oof_cat2)
# del oof_cat2, pred_cat2
# gc.collect()


# In[22]:


# train = h2o.H2OFrame(pd.concat([train_df, target], axis=1))                     
# test  = h2o.H2OFrame(test_df)
# print(train.shape, test.shape)

# gc.collect()

# x = train.columns
# y = "loan_default"
# x.remove(y)

# train[y] = train[y].asfactor()


# In[54]:


# from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator

# low_rank = H2OGeneralizedLowRankEstimator(k=5,#gamma_x=0.1, gamma_y=0.1,
#                                           regularization_x='l2',
#                                           regularization_y='l1',
#                                           seed=1234,
#                                           svd_method = "GramSVD",
#                                           max_iterations=5000
#                                           )
# low_rank.train(x, training_frame=train)


# In[23]:


# drop_cols = ['no_of_acc', 'NO.OF_INQUIRIES','new_acc_past_month',
#     'no_of_acc_overdue','loan_tenure_year','disbursal_month','Aadhar_flag',
#     'PAN_flag','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.SANCTIONED.AMOUNT',
#     'SEC.DISBURSED.AMOUNT','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.INSTAL.AMT',
#     'Passport_flag','VoterID_flag','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']


# pca_model.train(x, training_frame=train)


# In[24]:


# pca_train = pca_model.predict(train)
# pca_test  = pca_model.predict(test)

# pca_train = pca_train.as_data_frame()
# pca_test  = pca_test.as_data_frame()

# pca_train.shape, pca_test.shape


# In[ ]:




# dl_model.train(x, y, training_frame=train)#, validation_frame=test)
# model_path = h2o.save_model(model=dl_model, path="submission/stacking/", force=True)
# print(model_path)
# # var_df = pd.DataFrame(dl_model.varimp(), 
# #         columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])
# # print(var_df.shape)
# # var_df.head(10)
# pred = dl_model.predict(test)
# pred = pred.as_data_frame()
# pred_p1 = pred['p1'].values
# make_h20_prediction("dl_model_test", pred_p1, ts_unique_ids, sub)

# oof = dl_model.cross_validation_holdout_predictions()
# oof = oof.as_data_frame()
# oof_p1 = oof['p1'].values
# make_h20_prediction("dl_model_oof", oof_p1, tr_unique_ids, sub)

# del pred, pred_p1, oof, oof_p1
# gc.collect()

# # dl_model.varimp_plot()


# In[25]:


# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=5, weights='distance',
#                            algorithm='auto', leaf_size=30, 
#                            n_jobs=-1)
# knn.fit(pca_train.iloc[:-10000,:], target[:-10000])
# # knn.score(pca_train.iloc[-10000:,:], target[-10000:])
# roc_auc_score(target[-10000:], knn.predict_proba(pca_train.iloc[-10000:,:])[:,1])


# In[26]:


# from sklearn.neighbors import KDTree
# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# kdt = KDTree(X, leaf_size=30, metric='euclidean')
# kdt.query(X, k=2, return_distance=False)

# # pca_all = pd.concat([pca_train, pca_test], axis=0).reset_index(drop=True)

# # kdt = KDTree(pca_all, leaf_size=30, metric='euclidean')
# # nn = kdt.query(pca_all, 
# #           k=2, return_distance=False)


# In[52]:


gc.collect()


# In[54]:


import glob

int_feat_2way = glob.glob('store_data/new_feat/int_inter_2way_new/*')
feature_df = pd.DataFrame()

for file in int_feat_2way:
    print(file)
    feat_df = pd.read_csv(file)
    reduce_mem_usage_wo_print(feat_df)
    feature_df = pd.concat([feature_df, feat_df], axis=1)
    gc.collect()
feature_df.shape


# In[55]:


# del train, test
gc.collect()


# In[78]:


# model_xgb1, imp_xgb1, oof_xgb1, pred_xgb1 = run_cv_xgb(#train_df, target, test_df, 4)
#     pd.concat([train_df, feature_df.iloc[:target.shape[0]]], axis=1), target, 
#     pd.concat([test_df, feature_df.iloc[target.shape[0]:].reset_index(drop=True)], 
#               axis=1),
#     5)


# In[56]:


gc.collect()


# In[57]:


model_xgb2, imp_xgb2, oof_xgb2, pred_xgb2 = run_cv_xgb(#train_df, target, test_df, 4)
    feature_df.iloc[:target.shape[0]], target, 
    feature_df.iloc[target.shape[0]:].reset_index(drop=True), 6)

make_prediction("xgb_inter", pred_xgb2, ts_unique_ids, sub)
np.save('submission/stacking/oof_xgb_inter.npy',oof_xgb2)
del oof_xgb2, pred_xgb2
gc.collect()


# In[97]:


# imp = pd.DataFrame(data=[list(model_xgb2[2].get_fscore().keys()), 
#                    list(model_xgb2[2].get_fscore().values())]).T

# imp.columns = ['feat','imp']
# imp.sort_values(by='imp', ascending=False).head(50)


# In[58]:


try:
    imp_xgb2.imp = imp_xgb2.imp.astype('float')
    imp_xgb2 = imp_xgb2.groupby(['feature'])['imp'].mean()
    imp_xgb2 = pd.DataFrame(data=[imp_xgb2.index, imp_xgb2.values]).T
    imp_xgb2.columns=['feature','imp']
    imp_xgb2 = imp_xgb2.sort_values(by='imp', ascending=False).head(50)
    plt.figure(figsize=(12,25))
    plt.barh(imp_xgb2.feature, imp_xgb2.imp)
except:
    pass


# In[59]:


model_lgb2, imp_lgb2, oof_lgb2, pred_lgb2 = run_cv_lgb(#train_df, target, test_df)
    feature_df.iloc[:target.shape[0]], target, 
    feature_df.iloc[target.shape[0]:].reset_index(drop=True), 120)
make_prediction("lgb_inter", pred_lgb2, ts_unique_ids, sub)
np.save('submission/stacking/oof_lgb_inter.npy',oof_lgb2)
del oof_lgb2, pred_lgb2
gc.collect()


# In[60]:


try:
    imp_lgb2.imp = imp_lgb2.imp.astype('float')
    imp_lgb2 = imp_lgb2.groupby(['feature'])['imp'].mean()
    imp_lgb2 = pd.DataFrame(data=[imp_lgb2.index, imp_lgb2.values]).T
    imp_lgb2.columns=['feature','imp']
    imp_lgb2 = imp_lgb2.sort_values(by='imp', ascending=False).head(50)
    plt.figure(figsize=(12,25))
    plt.barh(imp_lgb2.feature, imp_lgb2.imp)
except:
    pass


# In[61]:


del feature_df
gc.collect()


# In[62]:


import glob

int_feat_3way = glob.glob('store_data/new_feat/int_inter_3way_new/*')
feature_df = pd.DataFrame()

for file in int_feat_3way[:2]:
    print(file)
    feat_df = pd.read_csv(file)
    reduce_mem_usage_wo_print(feat_df)
    feature_df = pd.concat([feature_df, feat_df], axis=1)
    gc.collect()

feature_df.shape


# In[63]:


model_xgb3, imp_xgb3, oof_xgb3, pred_xgb3 = run_cv_xgb(#train_df, target, test_df, 4)
    feature_df.iloc[:target.shape[0]], target, 
    feature_df.iloc[target.shape[0]:].reset_index(drop=True), 6)

make_prediction("xgb_inter_3way1", pred_xgb3, ts_unique_ids, sub)
np.save('submission/stacking/oof_xgb_inter_3way1.npy',oof_xgb3)
del oof_xgb3, pred_xgb3
gc.collect()


# In[64]:


model_lgb3, imp_lgb3, oof_lgb3, pred_lgb3 = run_cv_lgb(#train_df, target, test_df)
    feature_df.iloc[:target.shape[0]], target, 
    feature_df.iloc[target.shape[0]:].reset_index(drop=True), 120)
make_prediction("lgb_inter_3way1", pred_lgb3, ts_unique_ids, sub)
np.save('submission/stacking/oof_lgb_inter_3way1.npy',oof_lgb3)
del oof_lgb3, pred_lgb3
gc.collect()


# In[65]:


del feature_df
gc.collect()


# In[66]:


# int_feat_3way = glob.glob('store_data/new_feat/int_inter_3way_new/*')
feature_df = pd.DataFrame()

for file in int_feat_3way[2:]:
    print(file)
    feat_df = pd.read_csv(file)
    reduce_mem_usage_wo_print(feat_df)
    feature_df = pd.concat([feature_df, feat_df], axis=1)
    gc.collect()

feature_df.shape


# In[67]:


model_xgb4, imp_xgb4, oof_xgb4, pred_xgb4 = run_cv_xgb(#train_df, target, test_df, 4)
    feature_df.iloc[:target.shape[0]], target, 
    feature_df.iloc[target.shape[0]:].reset_index(drop=True), 6)

make_prediction("xgb_inter_3way2", pred_xgb4, ts_unique_ids, sub)
np.save('submission/stacking/oof_xgb_inter_3way2.npy',oof_xgb4)
del oof_xgb4, pred_xgb4
gc.collect()


# In[68]:


get_ipython().system('ls submission/stacking/')


# In[72]:


oof_xgb1 = np.load('submission/stacking/saturday/oof_xgb1.npy')
oof_xgb2 = np.load('submission/stacking/saturday/oof_xgb_inter.npy')
oof_xgb3 = np.load('submission/stacking/saturday/oof_xgb_inter_3way1.npy')
oof_xgb4 = np.load('submission/stacking/saturday/oof_xgb_inter_3way2.npy')

oof_lgb1 = np.load('submission/stacking/saturday/oof_lgb1.npy')
oof_lgb2 = np.load('submission/stacking/saturday/oof_lgb_inter.npy')
oof_lgb3 = np.load('submission/stacking/saturday/oof_lgb_inter_3way1.npy')

oof_ridge = np.load('submission/stacking/saturday/oof_ridge1.npy')
oof_logistic = np.load('submission/stacking/saturday/oof_logistic1.npy')

oof_dl = pd.read_csv('submission/stacking/saturday/dl_model_oof.csv')
oof_dl_deep = pd.read_csv('submission/stacking/saturday/dl_model_deep_oof.csv')

oof_drf = pd.read_csv('submission/stacking/saturday/drf_model_oof.csv')
oof_drf_deep = pd.read_csv('submission/stacking/saturday/drf_model_deep_oof.csv')

oof_gbm = pd.read_csv('submission/stacking/saturday/gbm_model_oof.csv')
oof_gbm_deep = pd.read_csv('submission/stacking/saturday/gbm_model_deep_oof.csv')

oof_glm = pd.read_csv('submission/stacking/saturday/glm_model_oof.csv')
oof_glm_bal = pd.read_csv('submission/stacking/saturday/glm_model_inter_oof.csv')
oof_glm_inter = pd.read_csv('submission/stacking/saturday/glm_model_bal_oof.csv')


# In[77]:


oofs_purane = np.column_stack([oof_xgb1,oof_xgb2, oof_xgb3, oof_xgb4,
                 oof_lgb1, oof_lgb2, oof_lgb3, oof_ridge, oof_logistic,
                 oof_dl.loan_default, oof_dl_deep.loan_default,
                 oof_drf.loan_default, oof_drf_deep.loan_default,
                 oof_gbm.loan_default, oof_gbm_deep.loan_default,
                 oof_glm.loan_default, oof_glm_bal.loan_default,
                 oof_glm_inter.loan_default])

oofs_purane = pd.DataFrame(data=oofs, columns=['xgb1', 'xgb2', 'xgb3', 'xgb4',
 'lgb1', 'lgb2', 'lgb3','ridge','logistic', 'dl','dll', 'drf','drfd','gbm','gbmd',
 'glm','glmd','glmdd'])

oofs_purane.columns = [col+'_p' for col in oofs_purane.columns]
oofs_purane.head()


# In[142]:


oofs = all_preds.copy()
corr = oofs.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[120]:


oofs1 = oofs[['xgb1', 'xgb4', 'lgb2', 'lgb3','logistic', 'dl','dll', 'drf',
              'drfd','gbmd','glmd']]
oofs.shape, oofs1.shape


# In[161]:


test_xgb1 = pd.read_csv('submission/stacking/xgb1.csv')
test_xgb2 = pd.read_csv('submission/stacking/xgb_inter.csv')
test_xgb4 = pd.read_csv('submission/stacking/xgb_inter_3way2.csv')

test_lgb1 = pd.read_csv('submission/stacking/lgb1.csv')
test_lgb2 = pd.read_csv('submission/stacking/lgb_inter.csv')
test_lgb3 = pd.read_csv('submission/stacking/lgb_inter_3way1.csv')

test_ridge = pd.read_csv('submission/stacking/ridge1.csv')
test_logistic = pd.read_csv('submission/stacking/logistic1.csv')

test_dl = pd.read_csv('submission/stacking/dl_model_deep_test.csv')
test_dl_deep = pd.read_csv('submission/stacking/dl_model_deep_test.csv')

test_drf = pd.read_csv('submission/stacking/drf_model_test.csv')
test_drf_deep = pd.read_csv('submission/stacking/drf_model_deep_test.csv')

test_gbm = pd.read_csv('submission/stacking/gbm_model_test.csv')
test_gbm_deep = pd.read_csv('submission/stacking/gbm_model_deep_test.csv')

test_glm_inter = pd.read_csv('submission/stacking/glm_model_bal_test.csv')


# In[165]:


tests = np.column_stack([test_xgb1.loan_default, test_xgb2.loan_default,
                         test_xgb4.loan_default, test_lgb1.loan_default,
                         test_lgb2.loan_default, test_lgb3.loan_default,
                         test_ridge.loan_default, test_logistic.loan_default, 
                         test_dl.loan_default, test_dl_deep.loan_default, 
                         test_drf.loan_default, test_drf_deep.loan_default, 
                         test_gbm.loan_default,test_gbm_deep.loan_default, 
                         test_glm_inter.loan_default,
                        ])

tests = pd.DataFrame(data=tests, columns=['xgb1','xgb2','xgb4','lgb1','lgb2','lgb3',
                                          'ridge','logistic','dl','dll','drf',
                                          'drfd','gbm','gbmd','glmd'])
tests.shape


# In[124]:


oofs1['mean'] = oofs1.mean(axis=1)
tests['mean'] = tests.mean(axis=1)

oofs1['std'] = oofs1.std(axis=1)
tests['std'] = tests.std(axis=1)

oofs1['ridge-logistic'] = oofs['ridge'] - oofs['logistic']
tests['ridge-logistic'] = tests['ridge'] - tests['logistic']

oofs1['xgb2-xgb1'] = oofs['xgb2'] - oofs['xgb1']
tests['xgb2-xgb1'] = tests['xgb2'] - tests['xgb1']

oofs1['glmd-logistic'] = oofs['glmd'] - oofs['logistic']
tests['glmd-logistic'] = tests['glmd'] - tests['logistic']

oofs1['gbm-gbmd'] = oofs['gbm'] - oofs['gbmd']
tests['gbm-gbmd'] = tests['gbm'] - tests['gbmd']

tests.drop(['gbm','xgb2','ridge'], axis=1, inplace=True)
oofs1.shape, tests.shape


# In[84]:


roc_auc_score(target, oofs1.mean(axis=1))


# In[81]:


roc_score = {}
for col in oofs.columns:
    roc_score[col] = roc_auc_score(target, oofs[col])
#     print(col.ljust(8), "==>", roc_auc_score(y_valid,preds[col]))
roc_score = pd.DataFrame(data=[roc_score.keys(), roc_score.values()]).T
roc_score.columns = ['model','roc']
roc_score['roc_norm'] = (roc_score.roc - roc_score.roc.min())/                         (roc_score.roc.max() - roc_score.roc.min())

plt.figure(figsize=(16,4))
g = sns.barplot(x='model',y='roc_norm',data=roc_score)

for index, row in roc_score.iterrows():
    g.text(row.name,row.roc_norm, round(row.roc,3), color='black', ha="center")


# In[128]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))/1024) for x in dir() if not x.startswith('_')     and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# In[130]:


del oof_dl, oof_dl_deep, oof_drf, oof_drf_deep, oof_gbm, oof_gbm_deep, oof_glm
del oof_glm_inter, oof_lgb1, oof_lgb2, oof_lgb3, oof_xgb1, oof_xgb2, oof_xgb3
gc.collect()


# In[125]:


train = h2o.H2OFrame(pd.concat([oofs1, target], axis=1))                     
test  = h2o.H2OFrame(tests)
print(train.shape, test.shape)

x = train.columns
y = "loan_default"
x.remove(y)

train[y] = train[y].asfactor()
gc.collect()


# In[126]:


gbm_model_meta = H2OGradientBoostingEstimator(  nfolds=3, seed=1234,
                                                fold_assignment = 'stratified',
                                                ntrees = 10000,
                                                learn_rate = 0.05,
                                                max_depth = 3,
                                                stopping_rounds = 5, 
                                                stopping_tolerance = 1e-3,
                                                stopping_metric = "AUC",
                                                sample_rate = 0.7,
                                                col_sample_rate = 0.7,
                                                keep_cross_validation_predictions=True
                                             )
gbm_model_meta.train(x, y, training_frame=train)#, validation_frame=test)
print(gbm_model_meta.accuracy())

# model_path = h2o.save_model(model=gbm_model_deep, path="submission/stacking/", force=True)
# print(model_path)

# pred = gbm_model_deep.predict(test)
# pred = pred.as_data_frame()
# pred_p1 = pred['p1'].values
# make_h20_prediction("gbm_model_deep_test", pred_p1, ts_unique_ids, sub)

# oof = gbm_model_deep.cross_validation_holdout_predictions()
# oof = oof.as_data_frame()
# oof_p1 = oof['p1'].values
# make_h20_prediction("gbm_model_deep_oof", oof_p1, tr_unique_ids, sub)

# del gbm_model_deep, pred, pred_p1, oof, oof_p1
# gc.collect()



gbm_model_meta.varimp_plot()

                                                             


# In[94]:


pred = gbm_model_meta.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("gbm_model_meta_test", pred_p1, ts_unique_ids, sub)


# In[114]:


# del dl_model_meta
gc.collect()


# In[131]:


drf_model_meta = H2ORandomForestEstimator( nfolds=3, seed=1234,
                                           keep_cross_validation_predictions=True,
                                           fold_assignment = 'stratified',
#                                            histogram_type = 'QuantilesGlobal',
#                                            categorical_encoding = 'eigen',
                                           stopping_metric = 'auc',
                                           ntrees = 200,
                                           balance_classes = True
                                           )
drf_model_meta.train(x, y, training_frame=train)#, validation_frame=test)
print(drf_model_meta.accuracy())


# model_path = h2o.save_model(model=drf_model_deep, path="submission/stacking/", force=True)
# print(model_path)

# pred = drf_model_deep.predict(test)
# pred = pred.as_data_frame()
# pred_p1 = pred['p1'].values
# make_h20_prediction("drf_model_deep_test", pred_p1, ts_unique_ids, sub)

# oof = drf_model_deep.cross_validation_holdout_predictions()
# oof = oof.as_data_frame()
# oof_p1 = oof['p1'].values
# make_h20_prediction("drf_model_deep_oof", oof_p1, tr_unique_ids, sub)

# del drf_model_deep, pred, pred_p1, oof, oof_p1
# gc.collect()




drf_model_meta.varimp_plot()


# In[104]:


pred = gbm_model_meta.predict(test)
pred = pred.as_data_frame()
pred_p1 = pred['p1'].values
make_h20_prediction("drf_model_meta_test", pred_p1, ts_unique_ids, sub)


# In[132]:


# for key in files.keys():
#     _submission[key + '_rank'] = _submission[key].rank()


# _submission['rank_sum'] = np.sum(
#         _submission[col] for col in _submission.columns if '_rank' in col)
# _submission['target'] = _submission['rank_sum']/(len(files) *
#         _submission.shape[0])

# # take the first (id) and last column (target)
# submission = _submission.iloc[:, [0, -1]]


# In[139]:


print(roc_auc_score(target, oofs1.mean(1)))


# In[140]:


reduce_mem_usage_wo_print(oofs1)
reduce_mem_usage_wo_print(oofs)


# In[142]:


# print(roc_auc_score(target, oofs1.mean(1)))
oofs2 = oofs1.rank()
oofs2 = oofs2/oofs2.max()
oofs2.max()


# In[156]:


oofs2 = oofs1[['xgb1','lgb3','drf','drfd','gbmd']].rank()
oofs2 = oofs2/oofs2.max()
print(roc_auc_score(target, oofs2.mean(1)))


# In[158]:


oofs2 = tests[['xgb1','lgb3','drf','drfd','gbmd']].rank()
oofs2 = oofs2/oofs2.max()

test_ranking = test_dl.copy()
test_ranking['loan_default'] = oofs2.mean(1)

test_ranking.to_csv('submission/stacking/ranking.csv', index=None)


# In[163]:


oofs2 = oofs[['xgb1','lgb1']].rank()
oofs2 = oofs2/oofs2.max()
print(roc_auc_score(target, oofs2.mean(1)))


# In[167]:


oofs2 = tests[['xgb1','lgb1']].rank()
oofs2 = oofs2/oofs2.max()

# test_ranking = test_dl.copy()
test_ranking['loan_default'] = oofs2.mean(1)

test_ranking.to_csv('submission/stacking/ranking1.csv', index=None)


# In[143]:


print(roc_auc_score(target, oofs1.mean(1)))
print(roc_auc_score(target, oofs2.mean(1)))


# In[53]:


# # make_prediction("cat_stratified", pred_cat1, ts_unique_ids, sub)

# model_ridge, oof_ridge, pred_ridge = run_cv_linear(#train_df, target, 
#                                                    #test_df,'logistic')
#     feature_df.iloc[:target.shape[0]], target, 
#     feature_df.iloc[target.shape[0]:].reset_index(drop=True),
#     'ridge')

# make_prediction("ridge_inter", pred_ridge, ts_unique_ids, sub)
# np.save('submission/stacking/oof_ridge_inter.npy',oof_ridge)
# del oof_ridge, pred_ridge
# gc.collect()


# In[276]:


# inter_cols = ['ltv','disbursed_amount','asset_cost','supplied_id', 'age(in year)',
#      'credit_hist_total_month', 'loan_tenure_total_month','PRI.CURRENT.BALANCE',
#      'PRI.DISBURSED.AMOUNT','PRI.SANCTIONED.AMOUNT','Bureau_desc']

# drop_cols = ['no_of_acc', 'NO.OF_INQUIRIES','new_acc_past_month',
#     'no_of_acc_overdue','loan_tenure_year','disbursal_month','Aadhar_flag',
#     'PAN_flag','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.SANCTIONED.AMOUNT',
#     'SEC.DISBURSED.AMOUNT','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.INSTAL.AMT',
#     'Passport_flag','VoterID_flag','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']

# glm_model_inter = H2OGeneralizedLinearEstimator(nfolds=2, seed=1234,
#                                                 keep_cross_validation_predictions=True,
#                                                 fold_assignment = 'stratified',
#                                                 ignored_columns = drop_cols,
#                                                 interactions = inter_cols,
#                                                 family="binomial", 
#                                                 lambda_search=True,
#                                                 balance_classes=True,
#                                                 remove_collinear_columns = True)
# glm_model_inter.train(x, y, training_frame=train, validation_frame=test)
# glm_model_inter.std_coef_plot()


# In[84]:



# drop_cols = ['no_of_acc', 'NO.OF_INQUIRIES','new_acc_past_month',
#     'no_of_acc_overdue','loan_tenure_year','disbursal_month','Aadhar_flag',
#     'PAN_flag','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.SANCTIONED.AMOUNT',
#     'SEC.DISBURSED.AMOUNT','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.INSTAL.AMT',
#     'Passport_flag','VoterID_flag','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']

# kmeans_model.train(x, y,training_frame=train, validation_frame=test)


# In[42]:



# drop_cols = ['no_of_acc', 'NO.OF_INQUIRIES','new_acc_past_month',
#     'no_of_acc_overdue','Aadhar_flag',#'loan_tenure_year','disbursal_month',
#     'PAN_flag','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.SANCTIONED.AMOUNT',
#     'SEC.DISBURSED.AMOUNT','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.INSTAL.AMT',
#     'Passport_flag','VoterID_flag','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']


# xgboost_model.train(x, y,training_frame=train, validation_frame=test)
# xgboost_model.varimp_plot()


# In[ ]:



# drop_cols = ['no_of_acc', 'NO.OF_INQUIRIES','new_acc_past_month',
#     'no_of_acc_overdue','Aadhar_flag',#'loan_tenure_year','disbursal_month',
#     'PAN_flag','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.SANCTIONED.AMOUNT',
#     'SEC.DISBURSED.AMOUNT','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.INSTAL.AMT',
#     'Passport_flag','VoterID_flag','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']

# xgboost_model1 = H2OXGBoostEstimator(ntrees = 10000,
#                                     max_bins=63,
#                                     col_sample_rate=0.5,
#                                     fold_assignment = 'stratified',
#                                     eta = 0.2,
#                                     keep_cross_validation_predictions=True, 
#                                     nfolds = 3,
#                                     ignored_columns = drop_cols,
#                                     seed=1234,
#                                     categorical_encoding = 'eigen',
#                                     grow_policy = 'depthwise', #lossguide(lgbm)
#                                     max_depth = 4,
#                                     min_child_weight = 5,
#                                     quiet_mode = False,
#                                     reg_lambda = 1.5,
#                                     sample_rate=0.5,
#                                     stopping_metric='auc',
#                                     stopping_rounds=15,
#                                     subsample=.5
#                                   )

# xgboost_model1.train(x, y,training_frame=train, validation_frame=test)
# xgboost_model1.varimp_plot()


# In[74]:



# drop_cols = ['no_of_acc', 'NO.OF_INQUIRIES','new_acc_past_month',
#     'no_of_acc_overdue','Aadhar_flag',#'loan_tenure_year','disbursal_month',
#     'PAN_flag','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.SANCTIONED.AMOUNT',
#     'SEC.DISBURSED.AMOUNT','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.INSTAL.AMT',
#     'Passport_flag','VoterID_flag','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']

# lgbm_model = H2OXGBoostEstimator(   ntrees = 10000,
#                                     max_bins=128,
#                                     col_sample_rate=0.7,
#                                     fold_assignment = 'stratified',
#                                     eta = 0.1,
#                                     keep_cross_validation_predictions=True, 
#                                     nfolds = 3,
#                                     ignored_columns = drop_cols,
#                                     seed=1234,
#                                     categorical_encoding = 'one_hot_explicit',
#                                     grow_policy = 'lossguide', #lossguide(lgbm)
#                                     max_depth = 5,
#                                     min_child_weight = 3,
#                                     quiet_mode = False,
#                                     reg_lambda = 1.5,
#                                     sample_rate=0.7,
#                                     stopping_metric='auc',
#                                     stopping_rounds=25,
#                                     subsample=.7
#                                   )

# lgbm_model.train(x, y,training_frame=train, validation_frame=test)
# lgbm_model.varimp_plot()


# In[180]:


# glm_model_poison = H2OGeneralizedLinearEstimator(balance_classes=True)
# glm_model_poison.train(x, y, training_frame=train, validation_frame=test)
# glm_model_poison.std_coef_plot()


# In[181]:


# glm_model_gauss = H2OGeneralizedLinearEstimator(family="gaussian", balance_classes=True)
# glm_model_gauss.train(x, y, training_frame=train, validation_frame=test)
# glm_model_gauss.std_coef_plot()


# In[41]:


# dl_model.cross_validation_holdout_predictions()


# In[40]:


# dl_pred  = dl_model.predict(test)
# drf_pred = drf_model.predict(test)
# gbm_pred = gbm_model.predict(test)
# glm_pred = glm_model.predict(test)

# dl_pred_deep  = dl_model_deep.predict(test)
# drf_pred_deep = drf_model_deep.predict(test)
# gbm_pred_deep = gbm_model_deep.predict(test)
# glm_pred_bal = glm_model_bal.predict(test)
# glm_pred_inter = glm_model_inter.predict(test)


# In[185]:


dl_pred  = dl_pred.as_data_frame()
drf_pred = drf_pred.as_data_frame()
gbm_pred = gbm_pred.as_data_frame()
glm_pred = glm_pred.as_data_frame()

dl_pred_deep  = dl_pred_deep.as_data_frame()
drf_pred_deep = drf_pred_deep.as_data_frame()
gbm_pred_deep = gbm_pred_deep.as_data_frame()
glm_pred_bal = glm_pred_bal.as_data_frame()


# In[186]:


preds = pd.concat([dl_pred.p1, drf_pred.p1, gbm_pred.p1, glm_pred.p1,
                  dl_pred_deep.p1, drf_pred_deep.p1, gbm_pred_deep.p1, 
                   glm_pred_bal.p1], axis=1)
preds.columns = ['dl','drf','gbm','glm', 'dl_deep','drf_deep','gbm_deep','glm_bal']
preds.shape


# In[187]:


import seaborn as sns
corr = preds.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[192]:


pred = lgb2.predict(x_valid, num_iteration=lgb2.best_iteration)
pred.shape, preds.shape


# In[197]:


preds = pd.concat([preds, pd.DataFrame(data=pred,columns=['lgbm'])], axis=1)
import seaborn as sns
corr = preds.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[209]:


roc_score = {}
for col in preds.columns:
    roc_score[col] = roc_auc_score(y_valid, preds[col])
#     print(col.ljust(8), "==>", roc_auc_score(y_valid,preds[col]))
roc_score = pd.DataFrame(data=[roc_score.keys(), roc_score.values()]).T
roc_score.columns = ['model','roc']
roc_score['roc_norm'] = (roc_score.roc - roc_score.roc.min())/                         (roc_score.roc.max() - roc_score.roc.min())


# In[221]:


plt.figure(figsize=(16,4))
g = sns.barplot(x='model',y='roc_norm',data=roc_score)

for index, row in roc_score.iterrows():
    g.text(row.name,row.roc_norm, round(row.roc,3), color='black', ha="center")


# In[198]:


train1 = h2o.H2OFrame(pd.concat([preds, target], axis=1))
# test  = h2o.H2OFrame(pd.concat([train_df, target], axis=1).drop(
#     ['disbursal_week', 'disbursal_day'], axis=1).iloc[valid_index])
# print(train.shape, test.shape)

# del train_df1, valid_df
gc.collect()

x = train1.columns
y = "loan_default"
x.remove(y)

# ignored_columns = ['UniqueID']
train1[y] = train1[y].asfactor()
# test[y]  = test[y].asfactor()

# aml = H2OAutoML(max_models=20, seed=1337, max_runtime_secs=14400, nfolds=3)
# aml.train(x = x, y = y,
#           training_frame = train,
#           leaderboard_frame = test)
# lb = aml.leaderboard
# lb.head(rows=lb.nrows)

gc.collect()


# In[222]:


gbm_model_meta = H2OGradientBoostingEstimator()
gbm_model_meta.train(x, y, training_frame=train1)#, validation_frame=test)
gbm_model_meta.varimp_plot()


# In[225]:


glm_model_meta = H2OGeneralizedLinearEstimator(family="binomial", 
                                              lambda_search=True,
                                              balance_classes=True)
glm_model_meta.train(x, y, training_frame=train1)#, validation_frame=test)
glm_model_meta.std_coef_plot()


# In[227]:


glm_model_bal


# In[155]:


from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

splits = 4
random_seed = 2019
features = train_df.columns
param = {
    'bagging_freq'           : 5,
    'bagging_fraction'       : 0.33,
    'boost_from_average'     : 'false',
    'boost'                  : 'gbdt',
    'feature_fraction'       : 0.3,
    'learning_rate'          : 0.01,
    'max_depth'              : -1,
    'metric'                 : 'auc',
    'min_data_in_leaf'       : 100,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves'             : 30,
    'num_threads'            : 4,
    'tree_learner'           : 'serial',
    'objective'              : 'binary',
    'verbosity'              : 1,
#     'lambda_l1'              : 0.001,
    'lambda_l2'              : 0.5
}   

n_splits = splits
num_round = 10000
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
oof_lgb = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

temp = train_df.apply(lambda x: pd.Series.value_counts(x).shape[0])
cat_columns = list(temp[temp<50].index)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print(trn_idx.shape, val_idx.shape)
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx],
                          feature_name=list(train_df.columns))#,
#                           categorical_feature=cat_columns)

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx],
                          feature_name=list(train_df.columns))#,
#                           categorical_feature=cat_columns) 

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], 
                    verbose_eval=50, early_stopping_rounds = 25)
    oof_lgb[val_idx] = clf.predict(train_df.iloc[val_idx][features], 
                               num_iteration=clf.best_iteration)
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
    
    print( "  auc = ", roc_auc_score(target.iloc[val_idx], oof_lgb[val_idx]) )
    print("="*60)

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof_lgb)))

# sub_df = pd.DataFrame({"ID_code":test_id})
# sub_df["target"] = predictions

# sub_df.columns = sub.columns
# sub_df.to_csv('submission/lightgbm_target_{}.csv'.format(file_path), index=None)


lgb_imp = pd.DataFrame(data=[clf.feature_name(), list(clf.feature_importance())]).T
lgb_imp.columns = ['feature','imp']
lgb_imp = lgb_imp.sort_values(by='imp', ascending=False)
plt.figure(figsize=(12,15))
plt.barh(lgb_imp.feature, lgb_imp.imp)

feature-engineering-final-stratified.py
Displaying feature-engineering-final-stratified-Copy1.py.